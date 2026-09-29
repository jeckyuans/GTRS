# Distance-sampled path head.
#
# HiP-AD predicts several spacings of the same geometric mode: a shared
# classification picks the mode, and each spacing has its own anchor residual.
# GTRS keeps its timed vocabulary for speed. At inference the selected
# vocabulary pose is moved along its own arc length onto this path, so the
# car keeps the scored speed profile and stays inside the expert corridor.

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SPATIAL_SPACINGS: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
N_POINTS = 6
N_MODES = 17
REF_SPACING = 1.0
INFER_SPACING = 2.0
# Constant-curvature fan, 1/m. At 12 m this is about +/- 6 m of lateral offset.
KAPPA_MAX = 0.08


def _spacing_name(spacing: float) -> str:
    return f"m{int(round(spacing * 10)):02d}"


def build_arc_anchors(
    spacings: Sequence[float] = SPATIAL_SPACINGS,
    n_points: int = N_POINTS,
    n_modes: int = N_MODES,
    kappa_max: float = KAPPA_MAX,
) -> Tensor:
    """Offsets of constant-curvature arcs. Index is shared across spacings.

    x is forward, y is left, matching the GTRS ego frame. Positive curvature
    turns left. The returned tensor is (F, K, N, 2).
    """
    kappas = torch.linspace(-kappa_max, kappa_max, n_modes)
    anchors = []
    for spacing in spacings:
        distance = torch.arange(1, n_points + 1, dtype=torch.float32) * float(spacing)
        kappa = kappas[:, None]
        arc = distance[None, :]
        straight = kappa.abs() < 1e-6
        theta = kappa * arc
        x = torch.where(straight, arc.expand_as(theta), torch.sin(theta) / kappa)
        y = torch.where(straight, torch.zeros_like(theta), (1.0 - torch.cos(theta)) / kappa)
        points = torch.stack([x, y], dim=-1)
        prev = torch.cat([points.new_zeros(n_modes, 1, 2), points[:, :-1]], dim=1)
        anchors.append(points - prev)
    return torch.stack(anchors, dim=0)


def sample_spatial_offsets(
    expert_xy: Tensor,
    spacing: float,
    n_points: int = N_POINTS,
) -> Tuple[Tensor, Tensor]:
    """Interpolate expert xy onto distances spacing, 2*spacing, ...

    expert_xy is (B, T, 2) in the current ego frame, without the origin.
    Returns offsets (B, N, 2) and a mask (B, N) that is 0 past the path end.
    """
    if expert_xy.ndim != 3 or expert_xy.shape[-1] < 2:
        raise ValueError("expert_xy must be (B, T, 2+)")
    xy = expert_xy[..., :2]
    batch, steps, _ = xy.shape
    origin = xy.new_zeros(batch, 1, 2)
    poly = torch.cat([origin, xy], dim=1)
    segment = torch.linalg.norm(poly[:, 1:] - poly[:, :-1], dim=-1)
    arc = torch.cumsum(torch.cat([segment.new_zeros(batch, 1), segment], dim=1), dim=1)

    target = xy.new_tensor([spacing * (i + 1) for i in range(n_points)])
    target = target[None].expand(batch, -1)
    index = torch.searchsorted(arc.contiguous(), target.contiguous()).clamp(1, steps)
    arc0 = arc.gather(1, index - 1)
    arc1 = arc.gather(1, index)
    point0 = poly.gather(1, (index - 1)[..., None].expand(batch, n_points, 2))
    point1 = poly.gather(1, index[..., None].expand(batch, n_points, 2))
    alpha = ((target - arc0) / (arc1 - arc0).clamp_min(1e-6)).clamp(0.0, 1.0)
    points = point0 + alpha[..., None] * (point1 - point0)

    valid = target <= (arc[:, -1:] + 1e-4)
    points = points * valid[..., None]
    prev = torch.cat([points.new_zeros(batch, 1, 2), points[:, :-1]], dim=1)
    offsets = (points - prev) * valid[..., None]
    return offsets, valid.to(dtype=xy.dtype)


_SPATIAL_GT = None


def load_spatial_gt(path: str):
    """Memory-map the prepared path file. points are (N, 4, 6, 2), mask is (N, 4, 6)."""
    global _SPATIAL_GT
    if _SPATIAL_GT is not None and _SPATIAL_GT[0] == path:
        return _SPATIAL_GT[1], _SPATIAL_GT[2]
    import numpy as np
    data = np.load(path, mmap_mode="r")
    index = {str(token): i for i, token in enumerate(data["tokens"])}
    _SPATIAL_GT = (path, data, index)
    return data, index


def lookup_spatial_points(tokens: Sequence[str], path: str) -> Tuple[Tensor, Tensor]:
    """Read prepared waypoints for these tokens. Missing tokens get an all-zero mask."""
    import numpy as np
    data, index = load_spatial_gt(path)
    points = data["points"]
    mask = data["mask"]
    batch_points = []
    batch_mask = []
    for token in tokens:
        row = index.get(str(token))
        if row is None:
            batch_points.append(torch.zeros(points.shape[1:], dtype=torch.float32))
            batch_mask.append(torch.zeros(mask.shape[1:], dtype=torch.float32))
        else:
            batch_points.append(torch.from_numpy(np.array(points[row], copy=True)))
            batch_mask.append(torch.from_numpy(np.array(mask[row], dtype=np.float32, copy=True)))
    return torch.stack(batch_points, dim=0), torch.stack(batch_mask, dim=0)


def points_to_offsets(points: Tensor, mask: Tensor) -> Tensor:
    """Absolute waypoints (B, F, N, 2) to per-step offsets. Invalid points stay zero."""
    prev = torch.cat([points.new_zeros(*points.shape[:-2], 1, 2), points[..., :-1, :]], dim=-2)
    return (points - prev) * mask[..., None]


def load_clustered_anchor_offsets(path: str) -> Tensor:
    """Absolute clustered waypoints (C, K, F, N, 2) to offsets of the same shape."""
    import numpy as np
    points = np.load(path)["points"].astype("float32")
    prev = np.concatenate([np.zeros_like(points[..., :1, :]), points[..., :-1, :]], axis=-2)
    return torch.from_numpy(points - prev)


class SpatialPathHead(nn.Module):
    """One query per anchor mode. Classification is shared; each spacing refines its own anchor."""

    def __init__(self, d_model: int, d_ffn: int, nhead: int = 8, anchor_path: str = ""):
        super().__init__()
        self.spacings = SPATIAL_SPACINGS
        self.n_points = N_POINTS
        if anchor_path:
            anchors = load_clustered_anchor_offsets(anchor_path)
            self.n_modes = int(anchors.shape[1])
        else:
            fan = build_arc_anchors().permute(1, 0, 2, 3).unsqueeze(0)
            anchors = fan
            self.n_modes = int(anchors.shape[1])
        self.mode_embed = nn.Embedding(self.n_modes, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ffn,
            dropout=0.0,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.cls_branch = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )
        self.reg_branches = nn.ModuleDict()
        for spacing in self.spacings:
            branch = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, N_POINTS * 2),
            )
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)
            self.reg_branches[_spacing_name(spacing)] = branch
        self.register_buffer("anchors", anchors, persistent=False)

    def forward(self, bev: Tensor, status: Tensor, command: Tensor = None) -> Dict[str, Tensor]:
        batch = bev.shape[0]
        if command is None:
            command = torch.zeros(batch, dtype=torch.long, device=bev.device)
        command = command.to(device=bev.device, dtype=torch.long).clamp(0, self.anchors.shape[0] - 1)
        bank = self.anchors[command]
        query = self.mode_embed.weight[None].expand(batch, -1, -1)
        query = query + status[:, None, :]
        query = self.decoder(query, bev)
        cls = self.cls_branch(query).squeeze(-1)

        offset_rows: List[Tensor] = []
        position_rows: List[Tensor] = []
        for index, spacing in enumerate(self.spacings):
            residual = self.reg_branches[_spacing_name(spacing)](query)
            residual = residual.view(batch, self.n_modes, self.n_points, 2)
            offsets = bank[:, :, index] + residual
            offset_rows.append(offsets)
            position_rows.append(torch.cumsum(offsets, dim=-2))

        offsets = torch.stack(offset_rows, dim=1)
        positions = torch.stack(position_rows, dim=1)
        mode = cls.argmax(dim=-1)
        infer_index = self.spacings.index(INFER_SPACING)
        chosen = positions[:, infer_index][torch.arange(batch, device=bev.device), mode]
        return {
            "spatial_cls": cls,
            "spatial_offsets": offsets,
            "spatial_positions": positions,
            "spatial_mode": mode,
            "spatial_path": chosen,
        }


def _positive_mode(positions: Tensor, gt_offsets: Tensor, mask: Tensor) -> Tensor:
    """Mode closest to the ground-truth path. positions is (B, K, N, 2)."""
    gt = torch.cumsum(gt_offsets, dim=-2)
    dist = torch.linalg.norm(positions - gt[:, None], dim=-1)
    dist = dist * mask[:, None]
    valid = mask.sum(dim=-1).clamp_min(1.0)
    return (dist.sum(dim=-1) / valid[:, None]).argmin(dim=-1)


def spatial_gt_batch(
    targets: Dict[str, Tensor],
    tokens: Sequence[str] = None,
    gt_path: str = None,
) -> Tuple[Tensor, Tensor]:
    """Prepared waypoints. Prefer tensors already in the batch, otherwise read the npz by token."""
    if "spatial_points" in targets and "spatial_mask" in targets:
        return targets["spatial_points"], targets["spatial_mask"]
    if not tokens or not gt_path:
        raise KeyError("spatial path loss reads spatial_points from the batch or spatial_gt_path by token")
    return lookup_spatial_points(tokens, gt_path)


def spatial_path_loss(
    predictions: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    spacings: Sequence[float] = SPATIAL_SPACINGS,
    tokens: Sequence[str] = None,
    gt_path: str = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Align every spacing to the reference spacing's nearest mode, then regress that mode."""
    if "spatial_positions" not in predictions:
        raise KeyError("spatial path is enabled but the model did not return spatial_positions")
    positions = predictions["spatial_positions"]
    offsets = predictions["spatial_offsets"]
    cls = predictions["spatial_cls"]
    gt_points, gt_mask = spatial_gt_batch(targets, tokens, gt_path)
    gt_points = gt_points.to(device=cls.device, dtype=cls.dtype)
    gt_mask = gt_mask.to(device=cls.device, dtype=cls.dtype)
    gt_offsets = points_to_offsets(gt_points, gt_mask)

    ref_index = list(spacings).index(REF_SPACING)
    ref_gt = gt_offsets[:, ref_index]
    ref_mask = gt_mask[:, ref_index]
    ref_valid = ref_mask.any(dim=-1)
    mode = _positive_mode(positions[:, ref_index], ref_gt, ref_mask)
    if not bool(ref_valid.any()):
        # A stopped batch still has to return a graph-connected zero.
        zero = cls.sum() * 0.0
        return zero, {"spatial_cls_loss": zero, "spatial_reg_loss": zero}

    cls_loss = F.cross_entropy(cls[ref_valid], mode[ref_valid])

    batch_index = torch.arange(cls.shape[0], device=cls.device)
    reg_loss = cls.new_zeros(())
    reg_terms = 0
    for index, _spacing in enumerate(spacings):
        pred = offsets[batch_index, index, mode]
        weight = gt_mask[:, index][..., None]
        denom = weight.sum().clamp_min(1.0)
        reg_loss = reg_loss + (pred - gt_offsets[:, index]).abs().mul(weight).sum() / denom
        reg_terms += 1
    reg_loss = reg_loss / reg_terms
    loss = cls_loss + reg_loss
    covered = ref_valid.float().mean()
    return loss, {
        "spatial_cls_loss": cls_loss,
        "spatial_reg_loss": reg_loss,
        "spatial_gt_covered": covered,
    }


def follow_spatial_path(trajectory: Tensor, path: Tensor) -> Tensor:
    """Place each timed pose at the same arc length on the spatial path.

    trajectory is (B, T, 3). path is (B, N, 2), the chosen waypoints, origin omitted.
    Past the last waypoint the last segment is extended, so a 12 m path still
    carries a longer 4 s plan without folding back onto the end point.
    """
    if trajectory.ndim != 3 or path.ndim != 3:
        raise ValueError("trajectory (B, T, 3) and path (B, N, 2) are required")
    batch, steps, _ = trajectory.shape
    if steps < 2:
        return trajectory
    xy = trajectory[..., :2]
    origin = xy.new_zeros(batch, 1, 2)
    poly = torch.cat([origin, path], dim=1)
    segment = poly[:, 1:] - poly[:, :-1]
    segment_len = torch.linalg.norm(segment, dim=-1)
    path_end = segment_len.sum(dim=-1)
    usable = path_end > 0.5

    step = torch.linalg.norm(xy[:, 1:] - xy[:, :-1], dim=-1)
    travel = torch.cumsum(torch.cat([step.new_zeros(batch, 1), step], dim=1), dim=1)
    arc = torch.cumsum(segment_len.clamp_min(0.0), dim=1)
    arc = torch.cat([arc.new_zeros(batch, 1), arc], dim=1)

    index = torch.searchsorted(arc.contiguous(), travel.contiguous()).clamp(1, poly.shape[1] - 1)
    arc0 = arc.gather(1, index - 1)
    point0 = poly.gather(1, (index - 1)[..., None].expand(batch, steps, 2))
    point1 = poly.gather(1, index[..., None].expand(batch, steps, 2))
    alpha = (travel - arc0) / (arc.gather(1, index) - arc0).clamp_min(1e-6)
    followed = point0 + alpha[..., None] * (point1 - point0)
    followed = torch.where(usable[:, None, None], followed, xy)

    forward = torch.cat([followed[:, 1:] - followed[:, :-1], followed[:, -1:] - followed[:, -2:-1]], dim=1)
    yaw = torch.atan2(forward[..., 1], forward[..., 0])
    yaw = torch.where(usable[:, None], yaw, trajectory[..., 2])
    return torch.cat([followed, yaw[..., None]], dim=-1)
