"""Candidate utility from frozen VoV features and independently valid labels.

GT geometry and PDM labels are used only by ``make_candidate_targets`` during
training. Deployment calls ``candidate_geometry`` with the predicted path.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .spatial_path import arc_length_deviation


class CandidateUtilityHead(nn.Module):
    """One row-independent scorer for the entire original trajectory vocabulary."""

    def __init__(self, query_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(query_dim + 6)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim + 6, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, candidate_features: Tensor, base_logits: Tensor, geometry: Tensor) -> Dict[str, Tensor]:
        if candidate_features.ndim != 3 or base_logits.shape != (*candidate_features.shape[:2], 3):
            raise ValueError("features must be [B,M,D] and base logits [B,M,3] in NC,DAC,EP order")
        if geometry.shape != (*candidate_features.shape[:2], 6):
            raise ValueError("geometry must be [B,M,6]")
        if not torch.isfinite(candidate_features).all() or not torch.isfinite(base_logits).all() or not torch.isfinite(geometry).all():
            raise ValueError("candidate utility inputs must be finite")
        residual = self.mlp(self.norm(torch.cat([candidate_features, geometry], dim=-1)))
        logits = torch.cat([residual[..., :1], base_logits + residual[..., 1:]], dim=-1)
        route, nc, dac, progress = logits.unbind(-1)
        # log of route * NC * DAC * min(progress / 0.8, 1), all in probability space.
        progress_log = (F.logsigmoid(progress) - torch.log(progress.new_tensor(0.8))).clamp(max=0.0)
        log_scores = F.logsigmoid(route) + F.logsigmoid(nc) + F.logsigmoid(dac) + progress_log
        return {"candidate_utility_logits": logits, "candidate_utility_log_scores": log_scores}


def candidate_geometry(candidates: Tensor, predicted_path: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """[B,M,6] route geometry from the *predicted* path, never GT.

    Feature order: maximum and mean equal-arc deviation / 4, shared arc / 48,
    shortfall / 48, candidate arc / 40, and minimum-overlap evaluability.
    """
    finite_path = torch.isfinite(predicted_path[..., :2]).all(-1)
    valid = finite_path if mask is None else (mask.bool() & finite_path)
    safe_path = torch.nan_to_num(predicted_path[..., :2], nan=0.0, posinf=0.0, neginf=0.0)
    safe_candidates = torch.nan_to_num(candidates[..., :2], nan=0.0, posinf=0.0, neginf=0.0)
    dev = arc_length_deviation(safe_candidates, safe_path, valid)
    features = torch.stack([
        dev.max_lateral / 4.0, dev.lateral / 4.0, dev.shared_arc / 48.0,
        dev.shortfall / 48.0, dev.candidate_arc / 40.0,
        dev.evaluable.to(dev.lateral.dtype),
    ], dim=-1)
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def deterministic_sample_indices(
    base_scores: Tensor,
    path_distance: Tensor,
    seed: int,
    k_high: int = 128,
    k_near: int = 128,
    k_uniform: int = 256,
) -> Tensor:
    """[B,K] original vocabulary indices, mixing score, path, and uniform rows.

    A caller can derive ``seed`` from a stable token hash. Selections are unique
    within each batch row; ties resolve by original index. No GT enters sampling.
    """
    if base_scores.ndim != 2 or path_distance.shape != base_scores.shape:
        raise ValueError("base_scores and path_distance must both be [B,M]")
    batch, n_vocab = base_scores.shape
    n_take = min(n_vocab, k_high + k_near + k_uniform)
    if min(k_high, k_near, k_uniform) < 0:
        raise ValueError("sample counts must be nonnegative")
    # Sampling is intentionally CPU and detached; sampled labels gather the same
    # original indices while the selected model scores retain their own gradient.
    score = torch.nan_to_num(base_scores.detach().float().cpu(), nan=-float("inf"))
    distance = torch.nan_to_num(path_distance.detach().float().cpu(), nan=float("inf"))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    rows = []
    for b in range(batch):
        chosen: list[int] = []
        seen: set[int] = set()
        for order, cap in ((torch.argsort(score[b], descending=True, stable=True), k_high),
                           (torch.argsort(distance[b], stable=True), k_near),
                           (torch.randperm(n_vocab, generator=generator), n_take)):
            if cap <= 0:
                continue
            added = 0
            for idx in order.tolist():
                if idx not in seen:
                    chosen.append(idx)
                    seen.add(idx)
                    added += 1
                    if len(chosen) == n_take or added == cap:
                        break
            if len(chosen) == n_take:
                break
        rows.append(chosen)
    return torch.tensor(rows, dtype=torch.long, device=base_scores.device)


def _project_to_expert(points: Tensor, expert: Tensor) -> tuple[Tensor, Tensor]:
    """Minimum perpendicular distance and arc coordinate on finite expert segments."""
    a, b = expert[:, :-1], expert[:, 1:]
    ab = b - a
    seg_len = torch.linalg.norm(ab, dim=-1)
    d = points[:, :, :, None, :] - a[:, None, None, :, :]
    u = ((d * ab[:, None, None]).sum(-1) / seg_len.square()[:, None, None].clamp_min(1e-8)).clamp(0.0, 1.0)
    foot = a[:, None, None] + u[..., None] * ab[:, None, None]
    dist = torch.linalg.norm(points[:, :, :, None] - foot, dim=-1)
    best = dist.argmin(-1, keepdim=True)
    nearest = dist.gather(-1, best).squeeze(-1)
    cumulative = F.pad(torch.cumsum(seg_len, dim=-1), (1, 0))
    s = cumulative[:, None, None, :-1] + u * seg_len[:, None, None]
    projected_s = s.gather(-1, best).squeeze(-1)
    return nearest, projected_s


def make_candidate_targets(
    candidates: Tensor,
    gt_path: Tensor,
    gt_mask: Tensor,
    foot_xy: Tensor,
    expert_s4: Tensor,
    expert_s4_valid: Tensor,
    nc_targets: Tensor,
    dac_targets: Tensor,
    nc_valid: Optional[Tensor] = None,
    dac_valid: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Labels for sampled vocabulary rows; all outputs are [B,M].

    The true foot is the expert polyline origin. Tail padding is ignored; a
    candidate is route-safe only if the expert covers its complete arc. A known
    crossing within the covered prefix is route-unsafe even with a short tail.
    """
    if candidates.ndim != 4 or gt_path.ndim != 3 or gt_mask.shape != gt_path.shape[:2]:
        raise ValueError("expected candidates [B,M,T,2+], GT [B,N,2], mask [B,N]")
    batch, n_candidate, n_step = candidates.shape[:3]
    if foot_xy.shape != (batch, 2):
        raise ValueError("foot_xy must be the true expert projection [B,2]")
    if nc_targets.shape != (batch, n_candidate) or dac_targets.shape != (batch, n_candidate):
        raise ValueError("NC/DAC targets must match sampled candidate rows")
    device = candidates.device
    dtype = candidates.dtype
    gt_ok = gt_mask.bool() & torch.isfinite(gt_path[..., :2]).all(-1)
    # Labels are a prefix; padding after the last valid knot is never an expert segment.
    contiguous = gt_ok.cumprod(dim=1).bool()
    valid_count = contiguous.sum(1)
    knot = torch.nan_to_num(gt_path[..., :2].to(dtype), nan=0.0, posinf=0.0, neginf=0.0)
    foot = torch.nan_to_num(foot_xy.to(dtype), nan=0.0, posinf=0.0, neginf=0.0)
    expert = torch.cat([foot[:, None], knot], dim=1)
    lengths = torch.linalg.norm(expert[:, 1:] - expert[:, :-1], dim=-1)
    lengths = torch.where(contiguous, lengths, torch.zeros_like(lengths))
    # Replacing padded knots by the last valid point keeps projection finite and
    # prevents a padded zero from manufacturing a shortcut segment.
    last = (valid_count - 1).clamp_min(0)
    last_point = knot.gather(1, last[:, None, None].expand(-1, 1, 2))
    expert[:, 1:] = torch.where(contiguous[..., None], knot, last_point)
    expert_len = lengths.sum(1)
    xy = torch.nan_to_num(candidates[..., :2], nan=0.0, posinf=0.0, neginf=0.0)
    origin = xy.new_zeros(batch, n_candidate, 1, 2)
    trace = torch.cat([origin, xy], dim=2)
    step_arc = torch.linalg.norm(trace[:, :, 1:] - trace[:, :, :-1], dim=-1)
    cand_arc = torch.cumsum(step_arc, dim=-1)
    finite_cand = torch.isfinite(candidates[..., :2]).all(-1).all(-1)
    valid_expert = torch.isfinite(foot_xy).all(-1) & (valid_count >= 2) & (expert_len >= 1.0)
    # Project both ego origin and every candidate sample onto actual expert
    # segments. Equal-arc distance would mislabel sharp curves and offroute feet.
    dist, projected_s = _project_to_expert(trace, expert)
    covered = torch.cat([torch.ones_like(cand_arc[..., :1], dtype=torch.bool), cand_arc <= expert_len[:, None, None] + 1e-3], 2)
    crossing = ((dist >= 4.0) & covered).any(2)
    complete = cand_arc[..., -1] <= expert_len[:, None] + 1e-3
    route_valid = valid_expert[:, None] & finite_cand & (complete | crossing)
    route_target = (~crossing).to(dtype)
    s_end = projected_s[..., -1]
    s_start = projected_s[..., 0]
    s4 = expert_s4.to(device=device, dtype=dtype).reshape(batch)
    s4_ok = expert_s4_valid.to(device=device).bool().reshape(batch) & torch.isfinite(s4) & (s4 > 1.0)
    # The current offroute s4 package measures from the rendered ego origin,
    # whereas this expert polyline measures from its true projected foot. Its
    # denominator includes lateral travel that the numerator correctly omits.
    # Do not learn a biased progress ratio from those frames until a foot-based
    # s4 label is available. Route supervision remains valid for those frames.
    centered_foot = torch.linalg.norm(foot_xy.to(dtype), dim=-1) <= 1e-3
    progress_valid = valid_expert[:, None] & finite_cand & complete & s4_ok[:, None] & centered_foot[:, None]
    progress_target = ((s_end - s_start).clamp_min(0.0) / s4[:, None].clamp_min(1.0)).clamp(0.0, 1.0)

    nc = nc_targets.to(device=device, dtype=dtype)
    dac = dac_targets.to(device=device, dtype=dtype)
    nc_mask = torch.isfinite(nc) & (nc >= 0.0) & (nc <= 1.0)
    dac_mask = torch.isfinite(dac) & (dac >= 0.0) & (dac <= 1.0)
    if nc_valid is not None:
        nc_mask &= nc_valid.to(device=device).bool()
    if dac_valid is not None:
        dac_mask &= dac_valid.to(device=device).bool()
    return {
        "route_target": route_target, "route_valid": route_valid,
        "nc_target": torch.nan_to_num(nc, nan=0.0), "nc_valid": nc_mask,
        "dac_target": torch.nan_to_num(dac, nan=0.0), "dac_valid": dac_mask,
        "progress_target": torch.nan_to_num(progress_target, nan=0.0), "progress_valid": progress_valid,
    }


def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    return (values * valid.to(values.dtype)).sum() / valid.sum().clamp_min(1)


def candidate_utility_loss(prediction: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Masked route, PDM, progress losses plus one known-pair ranking loss."""
    logits = prediction["candidate_utility_logits"]
    scores = prediction["candidate_utility_log_scores"]
    if logits.shape[:-1] != scores.shape or logits.shape[-1] != 4:
        raise ValueError("candidate utility prediction shapes do not match")
    terms: Dict[str, Tensor] = {}
    for i, name in enumerate(("route", "nc", "dac")):
        target, valid = targets[f"{name}_target"], targets[f"{name}_valid"].bool()
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        terms[f"{name}_loss"] = _masked_mean(F.binary_cross_entropy_with_logits(logits[..., i], safe_target, reduction="none"), valid)
    progress_target, progress_valid = targets["progress_target"], targets["progress_valid"].bool()
    safe_progress = torch.where(progress_valid, progress_target, torch.zeros_like(progress_target))
    terms["progress_loss"] = _masked_mean(F.smooth_l1_loss(torch.sigmoid(logits[..., 3]), safe_progress, reduction="none"), progress_valid)

    known = targets["route_valid"].bool() & targets["nc_valid"].bool() & targets["dac_valid"].bool() & progress_valid
    good = known & (targets["route_target"] >= 0.5) & (targets["nc_target"] >= 0.5) & (targets["dac_target"] >= 0.5) & (progress_target >= 0.8)
    bad = known & ((targets["route_target"] < 0.5) | (targets["nc_target"] < 0.5) | (targets["dac_target"] < 0.5))
    pair_valid = good.any(1) & bad.any(1)
    # Selection needs at least one known good row to beat the strongest known
    # bad row. BCE already supervises every good row separately.
    hard_positive = scores.masked_fill(~good, float("-inf")).amax(1)
    hard_negative = scores.masked_fill(~bad, float("-inf")).amax(1)
    # torch.where before softplus prevents inf-inf from poisoning autograd.
    gap = torch.where(pair_valid, hard_negative - hard_positive, torch.zeros_like(hard_positive))
    terms["ranking_loss"] = _masked_mean(F.softplus(gap), pair_valid)
    terms["loss"] = sum(terms.values())
    return terms
