# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

# Inference copy of SpatialPathHead from GTRS navsim/agents/gtrs_dense/spatial_path.py.
# Module and buffer names must stay identical so `_spatial_head.*` checkpoint keys load.
# No ground-truth reader. arc_length_deviation / route_constrained_pick match the GTRS
# copy: with GTRS_SPATIAL_PATH=1 the ORIGINAL candidate with the smallest max deviation
# from the predicted path is picked from the whole vocabulary, among rows within the
# score margin of the best base
# score with sigmoid(NC) and sigmoid(DAC) >= ROUTE_MIN_SAFETY_PROB, and executed
# unmodified, unless the scored pick is already within ROUTE_MIN_REFERENCE_DEV of the path
# or that row is not ROUTE_MIN_IMPROVEMENT closer.
# 方向可以保留，但这版还不是“训练一次就能修正执行路线”的完整方案。空间头学到的路径，
# 只有在它改变最终选中的那条【未改几何的】候选时，才接到了执行。

from __future__ import annotations

import math
from os import PathLike
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn

SPATIAL_SPACINGS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
# Points per spacing: 3 / 6 / 9 / 48 m, identical to GTRS. The rerank reads the 2.0 m
# tier. Tensors are padded to MAX_POINTS; slots past a spacing's count are zero.
SPATIAL_N_POINTS: tuple[int, ...] = (6, 6, 6, 24)
MAX_POINTS = max(SPATIAL_N_POINTS)
INFER_SPACING = 2.0
# The camera feature stitches l0[:, 416:-416] | f0 | r0[:, 416:-416] of 1920-wide
# images, so f0 spans pixels 1088..3008 of 4096 before the resize.
FRONT_VIEW_SPAN = (1088.0 / 4096.0, 3008.0 / 4096.0)
# Rerank guards, identical to GTRS. Uncalibrated: set only so that a stopped GTRS
# argmax is kept against a moving on-path candidate, not fit on data.
# Candidate/path overlap below MIN_SHARED_ARC metres is too short to judge laterally.
MIN_SHARED_ARC = 1.0
# A GTRS argmax that drives less than SHORT_ARC metres is compared only with other
# candidates below SHORT_ARC; a longer one with candidates within a factor ARC_RATIO.
SHORT_ARC = 2.0
ARC_RATIO = 1.5
# A candidate more than SCORE_MARGIN score units below the best score is never promoted.
SCORE_MARGIN = 1.0
# route_constrained_pick only promotes rows whose collision and drivable-area heads
# both give at least this probability. The stage-B oracle value, not tuned.
ROUTE_MIN_SAFETY_PROB = 0.9
# Deployed route_constrained_pick keeps the reference unless its own max deviation from
# the predicted path is at least ROUTE_MIN_REFERENCE_DEV metres and the chosen row's is
# lower by at least ROUTE_MIN_IMPROVEMENT metres, identical to GTRS. The predicted path
# is ~1.7 m (median) off the expert route, so an on-route reference can read 1-2 m off
# it. On the 128 corridor zeros at -2 s with the step-408 path this pair fixes 58 and
# changes 81 picks (ungated: 62 / 119; the old single 3.25 m gate: 55 / 74).
# GTRS_SPATIAL_RERANK_MIN_REF_DEV / GTRS_SPATIAL_RERANK_MIN_IMPROVEMENT override them
# in the driver.
ROUTE_MIN_REFERENCE_DEV = 2.0
ROUTE_MIN_IMPROVEMENT = 1.8


def front_view_columns(width: int) -> tuple[int, int]:
    start = int(math.floor(FRONT_VIEW_SPAN[0] * width))
    stop = int(math.ceil(FRONT_VIEW_SPAN[1] * width))
    return start, max(stop, start + 1)


def point_slots(n_points: tuple[int, ...] = SPATIAL_N_POINTS) -> Tensor:
    """(F, max(n_points)) bool: slot k of spacing f exists iff k < n_points[f]."""
    return torch.arange(max(n_points))[None, :] < torch.tensor(list(n_points))[:, None]


def load_clustered_anchor_offsets(path: str | PathLike[str]) -> Tensor:
    """Absolute clustered waypoints (C, K, F, MAX_POINTS, 2) to offsets; padded slots are zero."""
    with np.load(path) as archive:
        found = archive["n_points"] if "n_points" in archive.files else None
        if found is None or tuple(int(n) for n in found) != SPATIAL_N_POINTS:
            raise ValueError(
                f"{path} has n_points {None if found is None else tuple(found.tolist())}, "
                f"expected {SPATIAL_N_POINTS}"
            )
        points = archive["points"].astype("float32")
    prev = np.concatenate(
        [np.zeros_like(points[..., :1, :]), points[..., :-1, :]], axis=-2
    )
    return torch.from_numpy(points - prev) * point_slots()[..., None]


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
    index = (
        torch.searchsorted(
            arc.reshape(-1, n_poly).contiguous(),
            query.reshape(-1, n_query).contiguous(),
        )
        .view(*lead, n_query)
        .clamp(1, n_poly - 1)
    )
    arc0 = arc.gather(-1, index - 1)
    arc1 = arc.gather(-1, index)
    point0 = poly.gather(-2, (index - 1)[..., None].expand(*lead, n_query, 2))
    point1 = poly.gather(-2, index[..., None].expand(*lead, n_query, 2))
    alpha = ((query - arc0) / (arc1 - arc0).clamp_min(1e-6)).clamp(0.0, 1.0)
    return point0 + alpha[..., None] * (point1 - point0)


def arc_length_deviation(
    candidates: Tensor, path: Tensor, mask: Tensor | None = None
) -> PathDeviation:
    """Lateral deviation from the path on the shared arc, and coverage separately.

    candidates are (B, M, T, 2+) or (M, T, 2+) shared by the batch, without the ego
    origin. path is (B, N, 2) and mask (B, N) marks its valid points. Both polylines
    start at the origin and are compared at equal arc length, only on the shared arc
    min(candidate, valid path): the valid path points inside it plus its end point.
    The part of the path a candidate does not reach is shortfall (progress), not
    lateral error. Under MIN_SHARED_ARC of overlap a candidate is not evaluable.
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
    path_arc = torch.cumsum(
        torch.linalg.norm(path_poly[:, 1:] - path_poly[:, :-1], dim=-1), dim=1
    )
    path_arc0 = torch.cat([path_arc.new_zeros(batch, 1), path_arc], dim=1)
    path_len = (path_arc * mask).amax(dim=-1)

    cand_poly = torch.cat([xy.new_zeros(batch, n_cand, 1, 2), xy], dim=2)
    cand_seg = torch.linalg.norm(cand_poly[:, :, 1:] - cand_poly[:, :, :-1], dim=-1)
    cand_arc = torch.cat(
        [cand_seg.new_zeros(batch, n_cand, 1), torch.cumsum(cand_seg, dim=-1)], dim=-1
    )
    cand_len = cand_arc[..., -1]

    shared = torch.minimum(cand_len, path_len[:, None])
    point_arc = path_arc[:, None, :].expand(batch, n_cand, n_points)
    point_weight = mask[:, None, :] * (point_arc <= shared[..., None] + 1e-4).to(
        dtype=mask.dtype
    )
    # The shared-arc end is compared too, unless a valid path point already sits there.
    at_end = (
        point_weight
        * ((point_arc - shared[..., None]).abs() < 1e-3).to(dtype=mask.dtype)
    ).amax(dim=-1)
    weight = torch.cat([point_weight, (1.0 - at_end)[..., None]], dim=-1)

    query = torch.cat([point_arc, shared[..., None]], dim=-1)
    on_cand = _point_at_arc(cand_poly, cand_arc, query)
    path_end = _point_at_arc(
        path_poly[:, None].expand(batch, n_cand, n_points + 1, 2),
        path_arc0[:, None].expand(batch, n_cand, n_points + 1),
        shared[..., None],
    )
    on_path = torch.cat(
        [path[:, None].expand(batch, n_cand, n_points, 2), path_end], dim=2
    )
    dist = torch.linalg.norm(on_cand - on_path, dim=-1)
    lateral = (dist * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)
    shortfall = (path_len[:, None] - cand_len).clamp_min(0.0)
    return PathDeviation(
        lateral, shortfall, shared, cand_len, (dist * weight).amax(dim=-1)
    )


def longitudinally_comparable(candidate_arc: Tensor, reference_arc: Tensor) -> Tensor:
    """Below SHORT_ARC only other short candidates compare; else within ARC_RATIO."""
    near = (candidate_arc >= reference_arc / ARC_RATIO) & (
        candidate_arc <= reference_arc * ARC_RATIO
    )
    return torch.where(reference_arc < SHORT_ARC, candidate_arc < SHORT_ARC, near)


def route_constrained_pick(
    base_scores: Tensor,
    collision_prob: Tensor,
    drivable_prob: Tensor,
    candidates: Tensor,
    path: Tensor,
    mask: Tensor | None = None,
    reference: Tensor | None = None,
    score_margin: float = SCORE_MARGIN,
    min_safety_prob: float = ROUTE_MIN_SAFETY_PROB,
    arc_gate: bool = True,
    progress: Tensor | None = None,
    min_reference_deviation: float = 0.0,
    min_improvement: float = 0.0,
) -> tuple[Tensor, PathDeviation, Tensor]:
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
    to 9.8 m.

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
        if arc_gate
        else torch.ones_like(evaluable)
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


def walk_path_at_reference_arc(
    reference: Tensor, path: Tensor, mask: Tensor | None = None
) -> Tensor:
    """reference's arc-length schedule driven along path instead of its own geometry.

    reference (B, T, 3+) is an ego-frame trajectory (x forward, y left) without the
    origin; only its xy is read. path (B, N, 2) starts at the origin, and mask (B, N)
    marks its valid points, which must be a prefix. Pose t is the path point at the arc
    length reference has driven by step t, clamped to the valid path length: past the
    last valid point the pose stays on it, never extrapolated. Heading is the direction
    of the path segment under the pose, held from the previous pose while the pose sits
    at the origin or on a zero-length segment (0 before the first move). Returns
    (B, T, 3) as x, y, heading.
    """
    if path.ndim != 3 or reference.ndim != 3:
        raise ValueError("reference must be (B, T, 3+) and path (B, N, 2)")
    batch, n_points = path.shape[:2]
    path = path[..., :2]
    if mask is None:
        mask = path.new_ones(batch, n_points)
    mask = mask.to(dtype=path.dtype)
    xy = reference[..., :2].to(device=path.device, dtype=path.dtype)
    ref_poly = torch.cat([xy.new_zeros(batch, 1, 2), xy], dim=1)
    ref_arc = torch.cumsum(
        torch.linalg.norm(ref_poly[:, 1:] - ref_poly[:, :-1], dim=-1), dim=1
    )

    path_poly = torch.cat([path.new_zeros(batch, 1, 2), path], dim=1)
    seg = path_poly[:, 1:] - path_poly[:, :-1]
    seg_len = torch.linalg.norm(seg, dim=-1)
    path_arc = torch.cumsum(seg_len, dim=1)
    path_arc0 = torch.cat([path_arc.new_zeros(batch, 1), path_arc], dim=1)
    path_len = (path_arc * mask).amax(dim=-1, keepdim=True)

    query = torch.minimum(ref_arc, path_len)
    points = _point_at_arc(path_poly, path_arc0, query)

    segment = (
        torch.searchsorted(path_arc0.contiguous(), query.contiguous()).clamp(1, n_points)
        - 1
    )
    direction = seg.gather(1, segment[..., None].expand(-1, -1, 2))
    heading = torch.atan2(direction[..., 1], direction[..., 0])
    moved = (query > 1e-3) & (seg_len.gather(1, segment) > 1e-6)
    steps = torch.arange(query.shape[1], device=query.device).expand_as(query)
    last = torch.where(moved, steps, torch.full_like(steps, -1)).cummax(dim=1).values
    heading = torch.where(
        last >= 0, heading.gather(1, last.clamp_min(0)), torch.zeros_like(heading)
    )
    return torch.cat([points, heading[..., None]], dim=-1).to(dtype=reference.dtype)


def _spacing_name(spacing: float) -> str:
    return f"m{int(round(spacing * 10)):02d}"


def _linear_relu_ln(d_model: int, repeats: int) -> list[nn.Module]:
    layers: list[nn.Module] = []
    for _ in range(repeats):
        layers += [
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
        ]
    return layers


class SpatialPathHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        *,
        anchors: Tensor,
        nhead: int = 8,
        in_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.spacings = SPATIAL_SPACINGS
        self.n_points = SPATIAL_N_POINTS
        expected_tail = (len(self.spacings), MAX_POINTS, 2)
        if anchors.ndim != 5 or tuple(anchors.shape[2:]) != expected_tail:
            raise ValueError(
                f"spatial anchors must be (C, K, {expected_tail[0]}, "
                f"{expected_tail[1]}, 2) offsets for n_points {SPATIAL_N_POINTS}; "
                f"got {tuple(anchors.shape)}"
            )
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
        self.plan_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=0.0, batch_first=True
        )
        self.plan_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(inplace=True),
            nn.Linear(d_ffn, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.cls_branch = nn.Sequential(
            *_linear_relu_ln(d_model, 2), nn.Linear(d_model, 1)
        )
        self.reg_branches = nn.ModuleDict()
        for spacing, count in zip(self.spacings, self.n_points):
            self.reg_branches[_spacing_name(spacing)] = nn.Sequential(
                *_linear_relu_ln(d_model, 2), nn.Linear(d_model, count * 2)
            )
        self.register_buffer(
            "anchors", anchors.detach().clone().float() * slots[..., None]
        )
        self.register_buffer("point_slots", slots, persistent=False)

    def forward(
        self,
        front: Tensor,
        plan_query: Tensor | None = None,
        command: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch = front.shape[0]
        if command is None:
            command = torch.zeros(batch, dtype=torch.long, device=front.device)
        command = command.to(device=front.device, dtype=torch.long).clamp(
            0, self.anchors.shape[0] - 1
        )
        bank = self.anchors[command]
        slots = self.point_slots[..., None].to(dtype=bank.dtype)
        anchor_positions = torch.cumsum(bank, dim=-2) * slots

        query = self.front_encoder(front).flatten(1)
        query = query[:, None].expand(-1, self.n_modes, -1)
        query = query + self.anchor_encoder(
            anchor_positions[:, :, self.point_slots].flatten(2)
        )
        if plan_query is not None and plan_query.shape[1] > 0:
            context, _ = self.plan_attn(
                query, plan_query, plan_query, need_weights=False
            )
            query = self.plan_norm(query + context)
        query = self.ffn_norm(query + self.ffn(query))
        cls = self.cls_branch(query).squeeze(-1)

        width = self.point_slots.shape[1]
        offset_rows: list[Tensor] = []
        position_rows: list[Tensor] = []
        for index, (spacing, count) in enumerate(zip(self.spacings, self.n_points)):
            residual = self.reg_branches[_spacing_name(spacing)](query)
            residual = nn.functional.pad(
                residual.view(batch, self.n_modes, count, 2), (0, 0, 0, width - count)
            )
            offsets = bank[:, :, index] + residual
            offset_rows.append(offsets)
            position_rows.append(torch.cumsum(offsets, dim=-2) * slots[index])

        offsets = torch.stack(offset_rows, dim=1)
        positions = torch.stack(position_rows, dim=1)
        mode = cls.argmax(dim=-1)
        infer_index = self.spacings.index(INFER_SPACING)
        chosen = positions[:, infer_index, :, : self.n_points[infer_index]]
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
