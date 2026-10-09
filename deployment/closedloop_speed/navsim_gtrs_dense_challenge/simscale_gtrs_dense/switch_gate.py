"""Inference-only gate between the original VoV row and the fixed C0 route row."""

from __future__ import annotations

import torch
from torch import Tensor, nn

FEATURE_NAMES = (
    "ref_head_max_m",
    "c0_head_max_m",
    "head_improvement_m",
    "c0_arc_over_ref_arc",
    "ref_nc",
    "c0_nc",
    "ref_dac",
    "c0_dac",
    "ref_ep",
    "c0_ep",
    "c0_base_minus_ref_base",
    "predicted_path_len_m",
    "c0_path_evaluable",
)
GATE_THRESHOLD = 0.5


class SwitchGate(nn.Module):
    def __init__(self, feature_mean: Tensor, feature_std: Tensor):
        super().__init__()
        mean = torch.as_tensor(feature_mean, dtype=torch.float32)
        std = torch.as_tensor(feature_std, dtype=torch.float32)
        if mean.shape != (13,) or std.shape != (13,):
            raise ValueError("switch gate mean and std must each have 13 values")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("switch gate normalization must be finite with positive std")
        self.register_buffer("feature_mean", mean.clone())
        self.register_buffer("feature_std", std.clone())
        self.mlp = nn.Sequential(nn.Linear(13, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != 13:
            raise ValueError("switch gate input must be (B,13)")
        normalized = ((features - self.feature_mean) / self.feature_std).clamp(-8.0, 8.0)
        return self.mlp(normalized).squeeze(-1)


def switch_features(
    result: dict[str, Tensor], reference: Tensor, c0: Tensor, deviation, path: Tensor
) -> Tensor:
    """Build the fixed 13 features from current-frame predictions only."""
    ref = reference[:, None]
    candidate = c0[:, None]
    evaluable = deviation.evaluable
    max_lateral = deviation.max_lateral
    ref_dev = max_lateral.gather(1, ref)[:, 0]
    c0_dev = max_lateral.gather(1, candidate)[:, 0]
    c0_valid = evaluable.gather(1, candidate)[:, 0] & torch.isfinite(c0_dev)
    ref_dev = torch.where(torch.isfinite(ref_dev), ref_dev.clamp_max(20.0), ref_dev.new_full(ref_dev.shape, 20.0))
    c0_dev = torch.where(torch.isfinite(c0_dev), c0_dev.clamp_max(20.0), c0_dev.new_full(c0_dev.shape, 20.0))

    ref_arc = deviation.candidate_arc.gather(1, ref)[:, 0]
    c0_arc = deviation.candidate_arc.gather(1, candidate)[:, 0]
    arc_ratio = (c0_arc / ref_arc.clamp_min(1.0)).clamp(0.0, 4.0)

    def probability(name: str, indices: Tensor) -> Tensor:
        return result[name].gather(1, indices)[:, 0].sigmoid()

    origin = path.new_zeros(path.shape[0], 1, 2)
    polyline = torch.cat([origin, path], dim=1)
    path_length = torch.linalg.vector_norm(polyline[:, 1:] - polyline[:, :-1], dim=-1).sum(-1)
    # Training stores sigmoid head probabilities; reconstruct the same clipped
    # NC + DAC + 3 EP log score used for this feature.
    nc = result["no_at_fault_collisions"].sigmoid().clamp_min(1e-6)
    dac = result["drivable_area_compliance"].sigmoid().clamp_min(1e-6)
    ep = result["ego_progress"].sigmoid().clamp_min(1e-6)
    base = nc.log() + dac.log() + 3.0 * ep.log()
    feature_columns = (
        ref_dev,
        c0_dev,
        ref_dev - c0_dev,
        arc_ratio,
        probability("no_at_fault_collisions", ref),
        probability("no_at_fault_collisions", candidate),
        probability("drivable_area_compliance", ref),
        probability("drivable_area_compliance", candidate),
        probability("ego_progress", ref),
        probability("ego_progress", candidate),
        (base.gather(1, candidate)[:, 0] - base.gather(1, ref)[:, 0]).clamp(-20.0, 20.0),
        path_length,
        c0_valid.to(dtype=path.dtype),
    )
    features = torch.stack(feature_columns, dim=-1)
    if not torch.isfinite(features).all():
        raise ValueError("switch gate features contain non-finite values")
    return features
