# Distance-sampled path head.
#
# HiP-AD predicts several spacings of the same geometric mode: a shared
# classification picks the mode, and each spacing has its own anchor residual.
# GTRS stays frozen. At inference, with the head on, route_constrained_pick
# (stage B of docs/CLOSED_LOOP_TRAIN_PLAN.md) picks, from the whole scored pool,
# the ORIGINAL candidate with the smallest max deviation from the predicted path among those within
# spatial_rerank_score_margin of the best score with sigmoid(NC) and sigmoid(DAC)
# >= ROUTE_MIN_SAFETY_PROB, unless the reference is already within ROUTE_MIN_REFERENCE_DEV
# of the path or that candidate is not ROUTE_MIN_IMPROVEMENT closer. The chosen candidate
# is executed unmodified.
# rerank_by_spatial_path is the previous rule, kept as the oracle baseline.
#
# 方向可以保留，但这版还不是“训练一次就能修正执行路线”的完整方案。空间头学到的路径，
# 只有在它改变最终选中的那条【未改几何的】候选时，才接到了执行。

import math
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SPATIAL_SPACINGS: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
# Points per spacing: 3 / 6 / 9 / 48 m. The 2.0 m tier is what the rerank reads; 48 m
# covers the expert 4 s arc for ~99% of frames and is as far as the ~50 x 1 m source
# path labels for most frames. Tensors are padded to MAX_POINTS; slots past a
# spacing's count are zero and masked, never labels.
SPATIAL_N_POINTS: Tuple[int, ...] = (6, 6, 6, 24)
MAX_POINTS = max(SPATIAL_N_POINTS)
N_MODES = 17
INFER_SPACING = 2.0
# Per-step offset L1 is an auxiliary term; cumulative positions carry the geometry.
OFFSET_AUX_WEIGHT = 0.25
# Constant-curvature fan, 1/m. At 12 m this is about +/- 6 m of lateral offset.
KAPPA_MAX = 0.08
# The camera feature stitches l0[:, 416:-416] | f0 | r0[:, 416:-416] of 1920-wide
# images, so f0 spans pixels 1088..3008 of 4096 before the resize.
FRONT_VIEW_SPAN = (1088.0 / 4096.0, 3008.0 / 4096.0)
# Rerank guards. Uncalibrated: set only so that a stopped GTRS argmax is kept against
# a moving on-path candidate, not fit on data.
# Candidate/path overlap below MIN_SHARED_ARC metres is too short to judge laterally.
MIN_SHARED_ARC = 1.0
# A GTRS argmax that drives less than SHORT_ARC metres is compared only with other
# candidates below SHORT_ARC; a longer one with candidates within a factor ARC_RATIO.
SHORT_ARC = 2.0
ARC_RATIO = 1.5
# Default spatial_rerank_score_margin: a candidate more than this many score units
# below the best score is never promoted by geometry.
SCORE_MARGIN = 1.0
# route_constrained_pick only promotes rows whose collision and drivable-area heads
# both give at least this probability. The stage-B oracle value, not tuned.
ROUTE_MIN_SAFETY_PROB = 0.9
# Deployed route_constrained_pick keeps the reference unless its own max deviation from
# the predicted path is at least ROUTE_MIN_REFERENCE_DEV metres and the chosen row's is
# lower by at least ROUTE_MIN_IMPROVEMENT metres. The predicted path is ~1.7 m (median)
# off the expert route, so an on-route reference can read 1-2 m off it. On the 128
# corridor zeros at -2 s with the step-408 path this pair fixes 58 and changes 81 picks
# (ungated: 62 / 119; the old single 3.25 m gate: 55 / 74); among pairs fixing >= 58 it
# changes the fewest, and it is the largest such pair
# (scripts/evaluation/route_pick_improvement_gate_sweep.py).
ROUTE_MIN_REFERENCE_DEV = 2.0
ROUTE_MIN_IMPROVEMENT = 1.8


def front_view_columns(width: int) -> Tuple[int, int]:
    """Feature-grid columns [start, stop) covered by the front camera."""
    start = int(math.floor(FRONT_VIEW_SPAN[0] * width))
    stop = int(math.ceil(FRONT_VIEW_SPAN[1] * width))
    return start, max(stop, start + 1)


def _spacing_name(spacing: float) -> str:
    return f"m{int(round(spacing * 10)):02d}"


def point_slots(n_points: Sequence[int] = SPATIAL_N_POINTS) -> Tensor:
    """(F, max(n_points)) bool: slot k of spacing f exists iff k < n_points[f]."""
    width = max(n_points)
    return torch.arange(width)[None, :] < torch.tensor(list(n_points))[:, None]


def build_arc_anchors(
    spacings: Sequence[float] = SPATIAL_SPACINGS,
    n_points: Sequence[int] = SPATIAL_N_POINTS,
    n_modes: int = N_MODES,
    kappa_max: float = KAPPA_MAX,
) -> Tensor:
    """Offsets of constant-curvature arcs. Index is shared across spacings.

    x is forward, y is left, matching the GTRS ego frame. Positive curvature
    turns left. The returned tensor is (F, K, max(n_points), 2); padded slots are zero.
    """
    kappas = torch.linspace(-kappa_max, kappa_max, n_modes)
    width = max(n_points)
    anchors = []
    for spacing, count in zip(spacings, n_points):
        distance = torch.arange(1, count + 1, dtype=torch.float32) * float(spacing)
        kappa = kappas[:, None]
        arc = distance[None, :]
        straight = kappa.abs() < 1e-6
        theta = kappa * arc
        x = torch.where(straight, arc.expand_as(theta), torch.sin(theta) / kappa)
        y = torch.where(straight, torch.zeros_like(theta), (1.0 - torch.cos(theta)) / kappa)
        points = torch.stack([x, y], dim=-1)
        prev = torch.cat([points.new_zeros(n_modes, 1, 2), points[:, :-1]], dim=1)
        anchors.append(F.pad(points - prev, (0, 0, 0, width - count)))
    return torch.stack(anchors, dim=0)


def sample_spatial_offsets(
    expert_xy: Tensor,
    spacing: float,
    n_points: int = 6,
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


def _check_n_points(found, path: str) -> None:
    if found is None or tuple(int(n) for n in found) != SPATIAL_N_POINTS:
        raise ValueError(
            f"{path} has n_points {None if found is None else tuple(found.tolist())}, the head expects "
            f"{SPATIAL_N_POINTS}; files without n_points are the old 4 x 6 (12 m) contract")


def load_spatial_gt(path: str):
    """Load the prepared path file once per process.

    points are (N, 4, MAX_POINTS, 2), mask (N, 4, MAX_POINTS), n_points = SPATIAL_N_POINTS.
    """
    global _SPATIAL_GT
    if _SPATIAL_GT is not None and _SPATIAL_GT[0] == path:
        return _SPATIAL_GT[1], _SPATIAL_GT[2]
    import numpy as np
    # npz members are not memory-mapped; reading data["points"] per batch would decompress it every time.
    with np.load(path) as archive:
        data = {key: archive[key] for key in archive.files}
    _check_n_points(data.get("n_points"), path)
    expected = (len(SPATIAL_SPACINGS), MAX_POINTS)
    if tuple(data["points"].shape[1:]) != expected + (2,) or tuple(data["mask"].shape[1:]) != expected:
        raise ValueError(f"{path}: points {data['points'].shape} / mask {data['mask'].shape}, "
                         f"expected (N, {expected[0]}, {expected[1]}, 2) / (N, {expected[0]}, {expected[1]})")
    if data["mask"][:, ~point_slots().numpy()].any():
        raise ValueError(f"{path}: mask is set on padded slots past n_points")
    index = {str(token): i for i, token in enumerate(data["tokens"])}
    _SPATIAL_GT = (path, data, index)
    return data, index


def missing_spatial_tokens(tokens: Sequence[str], path: str) -> List[str]:
    """Tokens that have no row in the label file. A short tail (partial mask) is not missing."""
    _, index = load_spatial_gt(path)
    return [str(token) for token in tokens if str(token) not in index]


def lookup_spatial_points(tokens: Sequence[str], path: str) -> Tuple[Tensor, Tensor]:
    """Read prepared waypoints for these tokens. Missing tokens get an all-zero mask.

    Training goes through spatial_gt_batch, which refuses missing tokens instead.
    """
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
    """Absolute clustered waypoints (C, K, F, MAX_POINTS, 2) to offsets; padded slots are zero."""
    import numpy as np
    with np.load(path) as archive:
        _check_n_points(archive["n_points"] if "n_points" in archive.files else None, path)
        points = archive["points"].astype("float32")
    if tuple(points.shape[2:]) != (len(SPATIAL_SPACINGS), MAX_POINTS, 2):
        raise ValueError(f"{path}: anchor points {points.shape}, expected (C, K, "
                         f"{len(SPATIAL_SPACINGS)}, {MAX_POINTS}, 2)")
    prev = np.concatenate([np.zeros_like(points[..., :1, :]), points[..., :-1, :]], axis=-2)
    return torch.from_numpy(points - prev) * point_slots()[..., None]


def _linear_relu_ln(d_model: int, repeats: int) -> List[nn.Module]:
    layers: List[nn.Module] = []
    for _ in range(repeats):
        layers += [nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.LayerNorm(d_model)]
    return layers


class SpatialPathHead(nn.Module):
    """HiP-AD AlignPlan head on the frozen GTRS features.

    Every mode starts from the pooled front-camera feature plus its anchor
    embedding, as in PlanningInstanceBank.prepare_planning. The top scored
    GTRS vocabulary queries are attended as context. Classification is shared;
    each spacing regresses a residual onto its own copy of the mode's anchor.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        nhead: int = 8,
        anchor_path: str = "",
        in_channels: int = None,
    ):
        super().__init__()
        self.spacings = SPATIAL_SPACINGS
        self.n_points = SPATIAL_N_POINTS
        if anchor_path:
            anchors = load_clustered_anchor_offsets(anchor_path)
        else:
            anchors = build_arc_anchors().permute(1, 0, 2, 3).unsqueeze(0)
        self.n_modes = int(anchors.shape[1])
        slots = point_slots(self.n_points)
        in_channels = in_channels or d_model
        self.front_encoder = nn.Sequential(
            nn.Conv2d(in_channels, d_model, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.Conv2d(d_model, d_model, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.anchor_encoder = nn.Sequential(
            nn.Linear(sum(self.n_points) * 2, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
            *_linear_relu_ln(d_model, 1),
        )
        self.plan_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=True)
        self.plan_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(inplace=True),
            nn.Linear(d_ffn, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.cls_branch = nn.Sequential(*_linear_relu_ln(d_model, 2), nn.Linear(d_model, 1))
        self.reg_branches = nn.ModuleDict()
        for spacing, count in zip(self.spacings, self.n_points):
            branch = nn.Sequential(*_linear_relu_ln(d_model, 2), nn.Linear(d_model, count * 2))
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)
            self.reg_branches[_spacing_name(spacing)] = branch
        self.register_buffer("anchors", anchors * slots[..., None])
        self.register_buffer("point_slots", slots, persistent=False)

    def forward(
        self,
        front: Tensor,
        plan_query: Optional[Tensor] = None,
        command: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """front is the front-camera feature map (B, C, H, W); plan_query is (B, topk, D)."""
        batch = front.shape[0]
        if command is None:
            command = torch.zeros(batch, dtype=torch.long, device=front.device)
        command = command.to(device=front.device, dtype=torch.long).clamp(0, self.anchors.shape[0] - 1)
        bank = self.anchors[command]
        slots = self.point_slots[..., None].to(dtype=bank.dtype)
        anchor_positions = torch.cumsum(bank, dim=-2) * slots

        query = self.front_encoder(front).flatten(1)
        query = query[:, None].expand(-1, self.n_modes, -1)
        query = query + self.anchor_encoder(anchor_positions[:, :, self.point_slots].flatten(2))
        if plan_query is not None and plan_query.shape[1] > 0:
            context, _ = self.plan_attn(query, plan_query, plan_query, need_weights=False)
            query = self.plan_norm(query + context)
        query = self.ffn_norm(query + self.ffn(query))
        cls = self.cls_branch(query).squeeze(-1)

        width = self.point_slots.shape[1]
        offset_rows: List[Tensor] = []
        position_rows: List[Tensor] = []
        for index, (spacing, count) in enumerate(zip(self.spacings, self.n_points)):
            residual = self.reg_branches[_spacing_name(spacing)](query)
            residual = F.pad(residual.view(batch, self.n_modes, count, 2), (0, 0, 0, width - count))
            offsets = bank[:, :, index] + residual
            offset_rows.append(offsets)
            position_rows.append(torch.cumsum(offsets, dim=-2) * slots[index])

        offsets = torch.stack(offset_rows, dim=1)
        positions = torch.stack(position_rows, dim=1)
        mode = cls.argmax(dim=-1)
        infer_index = self.spacings.index(INFER_SPACING)
        chosen = positions[:, infer_index, :, :self.n_points[infer_index]]
        chosen = chosen[torch.arange(batch, device=front.device), mode]
        return {
            "spatial_cls": cls,
            "spatial_offsets": offsets,
            "spatial_positions": positions,
            "spatial_anchor_offsets": bank.permute(0, 2, 1, 3, 4),
            "spatial_anchor_positions": anchor_positions.permute(0, 2, 1, 3, 4),
            "spatial_mode": mode,
            "spatial_path": chosen,
        }


def _positive_mode(anchor_positions: Tensor, gt_offsets: Tensor, mask: Tensor) -> Tensor:
    """Nearest fixed anchor to the ground-truth path over every valid waypoint of every spacing.

    anchor_positions is (B, F, K, N, 2), gt_offsets (B, F, N, 2), mask (B, F, N).
    The label does not depend on the residual, so a mode cannot win by stretching
    itself onto the ground truth.
    """
    gt = torch.cumsum(gt_offsets, dim=-2)
    dist = torch.linalg.norm(anchor_positions - gt[:, :, None], dim=-1)
    dist = (dist * mask[:, :, None]).sum(dim=(1, 3))
    valid = mask.sum(dim=(1, 2)).clamp_min(1.0)
    return (dist / valid[:, None]).argmin(dim=-1)


def spatial_gt_batch(
    targets: Dict[str, Tensor],
    tokens: Sequence[str] = None,
    gt_path: str = None,
) -> Tuple[Tensor, Tensor]:
    """Prepared waypoints (B, F, N, 2) and mask (B, F, N).

    Prefer tensors already in the batch. Otherwise read the npz by token.
    A token that is not in the file stops training: a silent all-zero mask would
    give that sample no gradient and nobody would notice. Short tails (a partial
    mask) are ordinary labels.
    """
    if "spatial_points" in targets and "spatial_mask" in targets:
        return targets["spatial_points"], targets["spatial_mask"]
    if not tokens or not gt_path:
        raise KeyError("spatial path loss reads spatial_points from the batch or spatial_gt_path by token")
    missing = missing_spatial_tokens(tokens, gt_path)
    if missing:
        raise KeyError(
            f"{len(missing)} of {len(tokens)} tokens are not in {gt_path}, e.g. {missing[:3]}; "
            "run the preflight in navsim/planning/script/spatial_path_context.py"
        )
    return lookup_spatial_points(tokens, gt_path)


class PathDeviation(NamedTuple):
    """Per-candidate comparison with a spatial path, all (B, M) in metres.

    lateral: mean distance at equal arc length, only on the shared arc.
    shortfall: valid path length the candidate does not reach; not part of lateral.
    shared_arc: min(candidate arc length, valid path arc length).
    candidate_arc: arc length the candidate drives.
    max_lateral: max of the same distances; route_constrained_pick minimises this.
    """

    lateral: Tensor
    shortfall: Tensor
    shared_arc: Tensor
    candidate_arc: Tensor
    max_lateral: Tensor

    @property
    def evaluable(self) -> Tensor:
        return self.shared_arc >= MIN_SHARED_ARC


def _point_at_arc(poly: Tensor, arc: Tensor, query: Tensor) -> Tensor:
    """Points at arc lengths query (..., Q) on polylines poly (..., P, 2) with cumulative arc (..., P)."""
    lead = poly.shape[:-2]
    n_poly, n_query = poly.shape[-2], query.shape[-1]
    index = torch.searchsorted(
        arc.reshape(-1, n_poly).contiguous(), query.reshape(-1, n_query).contiguous(),
    ).view(*lead, n_query).clamp(1, n_poly - 1)
    arc0 = arc.gather(-1, index - 1)
    arc1 = arc.gather(-1, index)
    point0 = poly.gather(-2, (index - 1)[..., None].expand(*lead, n_query, 2))
    point1 = poly.gather(-2, index[..., None].expand(*lead, n_query, 2))
    alpha = ((query - arc0) / (arc1 - arc0).clamp_min(1e-6)).clamp(0.0, 1.0)
    return point0 + alpha[..., None] * (point1 - point0)


def arc_length_deviation(candidates: Tensor, path: Tensor, mask: Optional[Tensor] = None) -> PathDeviation:
    """Lateral deviation of each candidate from the path on their shared arc, and coverage separately.

    candidates are (B, M, T, 2+) or (M, T, 2+) shared by the batch, without the ego
    origin. path is (B, N, 2) and mask (B, N) marks its valid points; short tails are
    masked, not errors.

    Both polylines start at the origin and are compared at equal arc length, so time
    and speed do not enter. Only the shared arc min(candidate, valid path) is compared:
    the valid path points inside it plus the point at its end. A candidate that stops
    short of the path is not charged lateral error for the part it does not reach;
    that length is returned as shortfall, which is progress, not geometry. With less
    than MIN_SHARED_ARC of overlap (a stopped candidate, or a sample without a valid
    path) the lateral number compares almost nothing and the candidate is not
    evaluable; see PathDeviation.evaluable.
    """
    if path.ndim != 3:
        raise ValueError("path must be (B, N, 2)")
    batch, n_points = path.shape[:2]
    path = path[..., :2]
    if mask is None:
        mask = path.new_ones(batch, n_points)
    mask = mask.to(dtype=path.dtype)
    xy = candidates[..., :2].to(dtype=path.dtype)
    if xy.ndim == 3:
        xy = xy.unsqueeze(0).expand(batch, -1, -1, -1)
    n_cand = xy.shape[1]

    path_poly = torch.cat([path.new_zeros(batch, 1, 2), path], dim=1)
    path_arc = torch.cumsum(torch.linalg.norm(path_poly[:, 1:] - path_poly[:, :-1], dim=-1), dim=1)
    path_arc0 = torch.cat([path_arc.new_zeros(batch, 1), path_arc], dim=1)
    path_len = (path_arc * mask).amax(dim=-1)

    cand_poly = torch.cat([xy.new_zeros(batch, n_cand, 1, 2), xy], dim=2)
    cand_seg = torch.linalg.norm(cand_poly[:, :, 1:] - cand_poly[:, :, :-1], dim=-1)
    cand_arc = torch.cat([cand_seg.new_zeros(batch, n_cand, 1), torch.cumsum(cand_seg, dim=-1)], dim=-1)
    cand_len = cand_arc[..., -1]

    shared = torch.minimum(cand_len, path_len[:, None])
    point_arc = path_arc[:, None, :].expand(batch, n_cand, n_points)
    point_weight = mask[:, None, :] * (point_arc <= shared[..., None] + 1e-4).to(dtype=mask.dtype)
    # The shared-arc end is compared too, unless a valid path point already sits there.
    at_end = (point_weight * ((point_arc - shared[..., None]).abs() < 1e-3).to(dtype=mask.dtype)).amax(dim=-1)
    weight = torch.cat([point_weight, (1.0 - at_end)[..., None]], dim=-1)

    query = torch.cat([point_arc, shared[..., None]], dim=-1)
    on_cand = _point_at_arc(cand_poly, cand_arc, query)
    path_end = _point_at_arc(
        path_poly[:, None].expand(batch, n_cand, n_points + 1, 2),
        path_arc0[:, None].expand(batch, n_cand, n_points + 1),
        shared[..., None],
    )
    on_path = torch.cat([path[:, None].expand(batch, n_cand, n_points, 2), path_end], dim=2)
    dist = torch.linalg.norm(on_cand - on_path, dim=-1)
    lateral = (dist * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)
    shortfall = (path_len[:, None] - cand_len).clamp_min(0.0)
    return PathDeviation(lateral, shortfall, shared, cand_len, (dist * weight).amax(dim=-1))


def longitudinally_comparable(candidate_arc: Tensor, reference_arc: Tensor) -> Tensor:
    """Whether each candidate drives about as far as the reference (the GTRS argmax).

    Below SHORT_ARC the reference is stopped or creeping and only other candidates
    below SHORT_ARC compare; otherwise candidates within a factor ARC_RATIO do.
    """
    near = (candidate_arc >= reference_arc / ARC_RATIO) & (candidate_arc <= reference_arc * ARC_RATIO)
    return torch.where(reference_arc < SHORT_ARC, candidate_arc < SHORT_ARC, near)


def rerank_by_spatial_path(
    scores: Tensor,
    candidates: Tensor,
    path: Tensor,
    mask: Optional[Tensor] = None,
    weight: float = 1.0,
    score_margin: float = SCORE_MARGIN,
) -> Tuple[Tensor, PathDeviation]:
    """Index of max(scores - weight * lateral deviation) within the comparable pool.

    scores are (B, M) for candidates (M, T, 3) or (B, M, T, 3). The pool holds the
    GTRS argmax and candidates that are evaluable, longitudinally comparable to the
    argmax and within score_margin of its score. Coverage shortfall is never scored,
    so a long path cannot make a stopped argmax move. If the argmax itself is not
    evaluable it is kept. The caller executes candidates[index] as is.
    """
    deviation = arc_length_deviation(candidates, path, mask)
    scored = scores.argmax(dim=1)
    pick = scored[:, None]
    evaluable = deviation.evaluable
    pool = (
        evaluable
        & longitudinally_comparable(deviation.candidate_arc, deviation.candidate_arc.gather(1, pick))
        & (scores >= scores.gather(1, pick) - score_margin)
    )
    adjusted = (scores - weight * deviation.lateral.to(dtype=scores.dtype)).masked_fill(~pool, float("-inf"))
    index = torch.where(evaluable.gather(1, pick)[:, 0], adjusted.argmax(dim=1), scored)
    return index, deviation


def route_constrained_pick(
    base_scores: Tensor,
    collision_prob: Tensor,
    drivable_prob: Tensor,
    candidates: Tensor,
    path: Tensor,
    mask: Optional[Tensor] = None,
    reference: Optional[Tensor] = None,
    score_margin: float = SCORE_MARGIN,
    min_safety_prob: float = ROUTE_MIN_SAFETY_PROB,
    arc_gate: bool = True,
    progress: Optional[Tensor] = None,
    min_reference_deviation: float = 0.0,
    min_improvement: float = 0.0,
) -> Tuple[Tensor, PathDeviation, Tensor]:
    """Candidate with the smallest max path deviation among the near-best, safe rows of the whole pool.

    base_scores (B, M) must cover every candidate, before any top-k cut or speed bonus.
    collision_prob / drivable_prob (B, M) are sigmoid(NC) / sigmoid(DAC). reference (B,)
    is the row the caller executes otherwise; it defaults to the base_scores argmax.
    The pool is every row that is evaluable, longitudinally comparable to the
    reference (only if arc_gate), has base score >= max(base_scores) - score_margin
    and both probabilities >= min_safety_prob. The pool row with the smallest MAX
    distance to the path over the shared arc (PathDeviation.max_lateral) wins, ties to
    the lower index: the corridor is lost at the worst pose, so a low mean with one
    spike must not beat a smaller worst case. Shortfall is not scored. If the
    reference is not evaluable or the pool is empty the reference is
    kept. Returns the index, the deviation of every row and the pool size (B,). The
    caller executes candidates[index] as is.

    Without arc_gate, shorter rows compete and, being short, tend to have the smallest
    deviation: on the 0.850 corridor zeros at -2 s with the expert route extended to
    48 m this fixes 108/128 plans (79 with the gate), but only 25 keep >= 2/3 of the
    expert 4 s progress (58 with the gate); the median chosen arc drops from 17.3 m
    to 9.8 m (scripts/evaluation/route_pick_max_dev_replay.py).

    progress (B,) metres, if given, is the arc the gate is centred on instead of the
    reference row's arc (rows with a non-finite value keep the reference's). It only
    moves the gate; the reference fallback and the score/safety terms are unchanged.

    The reference is also kept when its own max_lateral is below min_reference_deviation
    metres: it already follows the path about as well as the path is known. 0 replaces
    it whenever the pool is non-empty; the deployed value is ROUTE_MIN_REFERENCE_DEV.

    It is kept as well unless the chosen row's max_lateral is lower than the reference's
    by at least min_improvement metres, so a row that is only slightly closer to an
    uncertain path does not replace it; with it on, a pool row farther than an
    out-of-pool reference never replaces it either. 0 disables this; the deployed value
    is ROUTE_MIN_IMPROVEMENT.
    """
    deviation = arc_length_deviation(candidates, path, mask)
    if reference is None:
        reference = base_scores.argmax(dim=1)
    pick = reference[:, None]
    evaluable = deviation.evaluable
    center = deviation.candidate_arc.gather(1, pick)
    if progress is not None:
        progress = progress.to(device=center.device, dtype=center.dtype).reshape(-1, 1)
        center = torch.where(torch.isfinite(progress), progress, center)
    comparable = (
        longitudinally_comparable(deviation.candidate_arc, center)
        if arc_gate else torch.ones_like(evaluable)
    )
    pool = (
        evaluable
        & comparable
        & (base_scores >= base_scores.amax(dim=1, keepdim=True) - score_margin)
        & (collision_prob >= min_safety_prob)
        & (drivable_prob >= min_safety_prob)
    )
    closest = deviation.max_lateral.masked_fill(~pool, float("inf")).argmin(dim=1)
    reference_dev = deviation.max_lateral.gather(1, pick)[:, 0]
    keep_reference = (
        ~evaluable.gather(1, pick)[:, 0]
        | ~pool.any(dim=1)
        | (reference_dev < min_reference_deviation)
    )
    if min_improvement > 0.0:
        # The reference need not be in the pool, so the chosen row can be farther from the path.
        closest_dev = deviation.max_lateral.gather(1, closest[:, None])[:, 0]
        keep_reference = keep_reference | (reference_dev - closest_dev < min_improvement)
    index = torch.where(keep_reference, reference, closest)
    return index, deviation, pool.sum(dim=1)


def spatial_path_loss(
    predictions: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    spacings: Sequence[float] = SPATIAL_SPACINGS,
    tokens: Sequence[str] = None,
    gt_path: str = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Classify the nearest fixed anchor; regress that anchor's cumulative positions onto the ground truth.

    The main geometric term is masked L1 on positions (cumsum of offsets), so a small
    per-step bias that accumulates along the path is paid at every later point.
    Per-step offset L1 stays as an auxiliary term with OFFSET_AUX_WEIGHT.
    """
    if "spatial_anchor_positions" not in predictions:
        raise KeyError("spatial path loss needs spatial_anchor_positions")
    anchor_positions = predictions["spatial_anchor_positions"]
    offsets = predictions["spatial_offsets"]
    positions = predictions["spatial_positions"]
    cls = predictions["spatial_cls"]
    gt_points, gt_mask = spatial_gt_batch(targets, tokens, gt_path)
    if tuple(gt_points.shape[1:]) != tuple(positions.shape[1:2]) + tuple(positions.shape[3:]):
        raise ValueError(f"spatial labels {tuple(gt_points.shape)} do not match the head's "
                         f"(B, {positions.shape[1]}, {positions.shape[3]}, 2); n_points={SPATIAL_N_POINTS}")
    gt_points = gt_points.to(device=cls.device, dtype=cls.dtype)
    slots = point_slots(SPATIAL_N_POINTS).to(device=cls.device, dtype=cls.dtype)
    gt_mask = gt_mask.to(device=cls.device, dtype=cls.dtype) * slots
    gt_offsets = points_to_offsets(gt_points, gt_mask)
    gt_positions = gt_points * gt_mask[..., None]

    valid = gt_mask.flatten(1).any(dim=-1)
    stats = {"spatial_gt_covered": valid.float().mean()}
    # Every rank must log the same keys (sync_dist all-reduces per key), so the
    # diagnostics are emitted even when this batch has no valid waypoint.
    stats.update(_spatial_diagnostics(predictions, positions, gt_points, gt_mask, spacings))
    mode = _positive_mode(anchor_positions, gt_offsets, gt_mask)
    if not bool(valid.any()):
        # Paths shorter than the first waypoint have no label.
        zero = cls.sum() * 0.0 + offsets.sum() * 0.0
        for key in ("spatial_cls_loss", "spatial_pos_loss", "spatial_offset_loss", "spatial_reg_loss"):
            stats[key] = zero
        return zero, stats

    cls_loss = F.cross_entropy(cls[valid], mode[valid])

    batch_index = torch.arange(cls.shape[0], device=cls.device)
    pos_loss = cls.new_zeros(())
    offset_loss = cls.new_zeros(())
    for index, _spacing in enumerate(spacings):
        weight = gt_mask[:, index][..., None]
        denom = weight.sum().clamp_min(1.0)
        pred_pos = positions[batch_index, index, mode]
        pos_loss = pos_loss + (pred_pos - gt_positions[:, index]).abs().mul(weight).sum() / denom
        pred_off = offsets[batch_index, index, mode]
        offset_loss = offset_loss + (pred_off - gt_offsets[:, index]).abs().mul(weight).sum() / denom
    pos_loss = pos_loss / len(spacings)
    offset_loss = offset_loss / len(spacings)
    reg_loss = pos_loss + OFFSET_AUX_WEIGHT * offset_loss
    loss = cls_loss + reg_loss
    stats["spatial_cls_loss"] = cls_loss
    stats["spatial_pos_loss"] = pos_loss
    stats["spatial_offset_loss"] = offset_loss
    stats["spatial_reg_loss"] = reg_loss
    return loss, stats


@torch.no_grad()
def _spatial_diagnostics(
    predictions: Dict[str, Tensor],
    positions: Tensor,
    gt_points: Tensor,
    gt_mask: Tensor,
    spacings: Sequence[float],
) -> Dict[str, Tensor]:
    """Masked errors of what the head and the planner actually pick, in metres.

    spatial_argmax_pos_error uses the mode the head selects, not the fixed-anchor
    positive. For the selected trajectory (and, in eval, the pre-rerank argmax as
    "scored") against the ground-truth 2.0 m path:
      *_traj_error          lateral deviation on the shared arc, over samples where
                            the trajectory is evaluable (>= MIN_SHARED_ARC overlap)
      *_coverage_shortfall  valid path length the trajectory does not reach; this is
                            progress, not route error, and is not inside *_traj_error
      *_traj_evaluable      fraction of samples with a valid path where it is evaluable
    A trajectory that merely starts moving lowers the shortfall, not the error.
    Training forward does not rerank, so there the selected trajectory is the GTRS
    argmax; in eval (validation) it is the reranked row.
    """
    gt_positions = gt_points * gt_mask[..., None]
    batch_index = torch.arange(positions.shape[0], device=positions.device)
    picked = positions[batch_index, :, predictions["spatial_mode"]]
    err = torch.linalg.norm(picked - gt_positions, dim=-1)
    stats = {"spatial_argmax_pos_error": (err * gt_mask).sum() / gt_mask.sum().clamp_min(1.0)}
    infer_index = list(spacings).index(INFER_SPACING)
    path_valid = gt_mask[:, infer_index].any(dim=-1).to(dtype=positions.dtype)
    for key, prefix in (("trajectory", "spatial_selected"), ("trajectory_scored", "spatial_scored")):
        if key not in predictions:
            continue
        dev = arc_length_deviation(
            predictions[key][:, None].to(device=positions.device, dtype=positions.dtype),
            gt_points[:, infer_index],
            gt_mask[:, infer_index],
        )
        evaluable = dev.evaluable[:, 0].to(dtype=positions.dtype) * path_valid
        stats[f"{prefix}_traj_error"] = (dev.lateral[:, 0] * evaluable).sum() / evaluable.sum().clamp_min(1.0)
        stats[f"{prefix}_coverage_shortfall"] = (
            (dev.shortfall[:, 0] * path_valid).sum() / path_valid.sum().clamp_min(1.0))
        stats[f"{prefix}_traj_evaluable"] = evaluable.sum() / path_valid.sum().clamp_min(1.0)
    return stats


def follow_spatial_path(trajectory: Tensor, path: Tensor) -> Tensor:
    """Place each timed pose at the same arc length on the spatial path.

    trajectory is (B, T, 3) and does not include the ego origin. path is (B, N, 2).
    Travel starts at the distance from the origin to the first pose. Past the last
    waypoint the last segment is extended in a straight line, which is not a lane.

    Not used to build the executed trajectory: after scoring it would move poses the
    GTRS score never saw, and a 48 m path is extrapolated past its end. Kept for tests
    and offline inspection only; closed loop reranks with rerank_by_spatial_path.
    """
    if trajectory.ndim != 3 or path.ndim != 3:
        raise ValueError("trajectory (B, T, 3) and path (B, N, 2) are required")
    batch, steps, _ = trajectory.shape
    if steps < 1:
        return trajectory
    xy = trajectory[..., :2]
    origin = xy.new_zeros(batch, 1, 2)
    poly = torch.cat([origin, path], dim=1)
    segment = poly[:, 1:] - poly[:, :-1]
    segment_len = torch.linalg.norm(segment, dim=-1)
    path_end = segment_len.sum(dim=-1)
    usable = path_end > 0.5

    origin_step = torch.linalg.norm(xy[:, :1], dim=-1)
    later = torch.linalg.norm(xy[:, 1:] - xy[:, :-1], dim=-1) if steps > 1 else xy.new_zeros(batch, 0)
    travel = torch.cumsum(torch.cat([origin_step, later], dim=1), dim=1)
    arc = torch.cumsum(segment_len.clamp_min(0.0), dim=1)
    arc = torch.cat([arc.new_zeros(batch, 1), arc], dim=1)

    index = torch.searchsorted(arc.contiguous(), travel.contiguous()).clamp(1, poly.shape[1] - 1)
    arc0 = arc.gather(1, index - 1)
    point0 = poly.gather(1, (index - 1)[..., None].expand(batch, steps, 2))
    point1 = poly.gather(1, index[..., None].expand(batch, steps, 2))
    alpha = (travel - arc0) / (arc.gather(1, index) - arc0).clamp_min(1e-6)
    followed = point0 + alpha[..., None] * (point1 - point0)
    followed = torch.where(usable[:, None, None], followed, xy)

    if steps == 1:
        forward = followed
    else:
        forward = torch.cat(
            [followed[:, 1:] - followed[:, :-1], followed[:, -1:] - followed[:, -2:-1]],
            dim=1,
        )
    yaw = torch.atan2(forward[..., 1], forward[..., 0])
    yaw = torch.where(usable[:, None], yaw, trajectory[..., 2])
    return torch.cat([followed, yaw[..., None]], dim=-1)
