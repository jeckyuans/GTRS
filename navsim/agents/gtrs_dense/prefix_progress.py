"""Timed candidate-prefix progress, independent of route and risk heads.

Inference uses only frozen candidate features, predicted-path geometry, row
geometry and current velocity. Expert future coordinates are loss-only data.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


PREFIX_INDICES = (4, 9, 19, 39)  # 0.5, 1, 2, 4 seconds at the 0.1 s grid
MIN_EXPERT_ADVANCE_M = 0.1  # numerical resolution guard, fixed before training


def prefix_candidate_arcs(candidates: Tensor) -> Tensor:
    """Cumulative candidate XY arc in metres at the four fixed time horizons."""
    if candidates.ndim != 4 or candidates.shape[2] != 40 or candidates.shape[-1] < 2:
        raise ValueError("candidates must be [B,M,40,2+]")
    xy = candidates[..., :2]
    if not torch.isfinite(xy).all():
        raise ValueError("candidate coordinates must be finite")
    previous = torch.cat([xy.new_zeros(*xy.shape[:2], 1, 2), xy[..., :-1, :]], dim=2)
    cumulative = torch.linalg.norm(xy - previous, dim=-1).cumsum(-1)
    return cumulative[..., list(PREFIX_INDICES)]


class PrefixProgressHead(nn.Module):
    """Predict fractional expert-route progress at each fixed time horizon."""

    def __init__(self, query_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        input_dim = query_dim + 6 + 4 + 2
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 4))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        candidate_features: Tensor,
        geometry: Tensor,
        candidate_arcs: Tensor,
        ego_velocity: Tensor,
    ) -> Dict[str, Tensor]:
        if candidate_features.ndim != 3:
            raise ValueError("candidate features must be [B,M,D]")
        batch, rows = candidate_features.shape[:2]
        if geometry.shape != (batch, rows, 6) or candidate_arcs.shape != (batch, rows, 4):
            raise ValueError("geometry and prefix arcs must be [B,M,6] and [B,M,4]")
        if ego_velocity.shape != (batch, 2):
            raise ValueError("ego velocity must be [B,2], from status columns 4:6")
        if not all(torch.isfinite(x).all() for x in (candidate_features, geometry, candidate_arcs, ego_velocity)):
            raise ValueError("prefix progress inputs must be finite")
        velocity = ego_velocity[:, None, :].expand(batch, rows, 2)
        x = torch.cat([candidate_features, geometry, candidate_arcs / 40.0, velocity], dim=-1)
        logits = self.mlp(self.norm(x))
        # Frozen P2/R1 deploys the continuous geometric mean. The teacher
        # remains an unsaturated progress fraction; saturation is not a score
        # transform in this R1 lineage.
        log_factors = F.logsigmoid(logits)
        return {
            "prefix_progress_logits": logits,
            "prefix_progress_log_score": log_factors.mean(-1),
        }


def prefix_utility_log_score(utility_logits: Tensor, prefix_progress_log_score: Tensor) -> Tensor:
    """Frozen U1 route/NC/DAC terms plus the new timed progress term."""
    if utility_logits.shape[-1] != 4 or utility_logits.shape[:-1] != prefix_progress_log_score.shape:
        raise ValueError("expected utility logits [B,M,4] and prefix score [B,M]")
    return F.logsigmoid(utility_logits[..., :3]).sum(-1) + prefix_progress_log_score


def make_prefix_targets(
    candidates: Tensor,
    expert_future: Tensor,
    expert_mask: Tensor,
    offroute: Tensor,
) -> Dict[str, Tensor]:
    """Time-aligned GT progress fractions with explicit per-horizon validity.

    Both candidate and expert use 40 positions on the 0.1 s grid. A candidate
    endpoint is projected onto the expert polyline from now through the same
    horizon; a faster candidate saturates at 1. The route head independently
    handles lateral validity. Rendered offroute frames lack an exact-pose timed
    expert label and contribute no prefix loss.
    """
    if candidates.ndim != 4 or candidates.shape[2] != 40 or candidates.shape[-1] < 2:
        raise ValueError("candidates must be [B,M,40,2+]")
    batch, rows = candidates.shape[:2]
    if expert_future.shape != (batch, 40, 2) or expert_mask.shape != (batch, 40):
        raise ValueError("expert future and mask must be [B,40,2] and [B,40]")
    if offroute.shape != (batch,):
        raise ValueError("offroute must be [B]")
    dtype = candidates.dtype
    device = candidates.device
    cand = torch.nan_to_num(candidates[..., :2], nan=0.0, posinf=0.0, neginf=0.0)
    expert_ok = expert_mask.bool() & torch.isfinite(expert_future).all(-1)
    expert_ok = expert_ok.cumprod(-1).bool()
    expert = torch.nan_to_num(expert_future.to(dtype=dtype, device=device), nan=0.0, posinf=0.0, neginf=0.0)
    poly = torch.cat([expert.new_zeros(batch, 1, 2), expert], dim=1)
    segments = poly[:, 1:] - poly[:, :-1]
    lengths = torch.linalg.norm(segments, dim=-1)
    arc = torch.cat([lengths.new_zeros(batch, 1), lengths.cumsum(-1)], dim=1)
    labels = []
    masks = []
    candidate_finite = torch.isfinite(candidates[..., :2]).all(-1).cumprod(-1).bool()
    for k in PREFIX_INDICES:
        # Only the expert prefix through this horizon is used; no later GT can
        # leak into an earlier label. Orthogonal projection handles curved paths.
        a, ab = poly[:, :k + 1], segments[:, :k + 1]
        seglen = lengths[:, :k + 1]
        point = cand[:, :, k]
        delta = point[:, :, None, :] - a[:, None]
        u = ((delta * ab[:, None]).sum(-1) / seglen[:, None].square().clamp_min(1e-8)).clamp(0.0, 1.0)
        projection = a[:, None] + u[..., None] * ab[:, None]
        distance = torch.linalg.norm(point[:, :, None] - projection, dim=-1)
        best = distance.argmin(-1, keepdim=True)
        projected_arc = arc[:, None, :k + 1] + u * seglen[:, None]
        progress = projected_arc.gather(-1, best).squeeze(-1)
        horizon_arc = arc[:, k + 1]
        valid = (expert_ok[:, k, None] & (horizon_arc[:, None] > MIN_EXPERT_ADVANCE_M)
                 & candidate_finite[:, :, k] & ~offroute.bool()[:, None])
        labels.append((progress / horizon_arc[:, None].clamp_min(MIN_EXPERT_ADVANCE_M)).clamp(0.0, 1.0))
        masks.append(valid)
    target = torch.stack(labels, -1)
    valid = torch.stack(masks, -1)
    return {
        "prefix_progress_target": torch.where(valid, target, torch.zeros_like(target)),
        "prefix_progress_valid": valid,
    }


def prefix_progress_loss(prediction: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Soft-label BCE averaged only over known candidate/horizon pairs."""
    logits = prediction["prefix_progress_logits"]
    target = targets["prefix_progress_target"]
    valid = targets["prefix_progress_valid"].bool()
    if logits.shape != target.shape or logits.shape != valid.shape or logits.shape[-1] != 4:
        raise ValueError("prefix logits, targets and mask must all be [B,M,4]")
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    element = F.binary_cross_entropy_with_logits(logits, safe_target, reduction="none")
    loss = (element * valid.to(element.dtype)).sum() / valid.sum().clamp_min(1)
    return {"loss": loss, "prefix_progress_loss": loss}
