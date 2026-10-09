# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import MemoryEffTransformer
from .backbone import TransfuserBackbone, VoVBackbone
from .candidate_utility import CandidateUtilityHead, candidate_geometry
from .prefix_progress import PrefixProgressHead, prefix_candidate_arcs, prefix_utility_log_score
from .inference_cache import InferenceTensorCache
from .config import GTRSDenseConfig
from .spatial_path import (
    ROUTE_MIN_IMPROVEMENT,
    ROUTE_MIN_REFERENCE_DEV,
    SCORE_MARGIN,
    SpatialPathHead,
    front_view_columns,
    route_constrained_pick,
)
from .switch_gate import GATE_THRESHOLD, switch_features

RELEASE_SCORER = "release"
NC_DAC_EP_SCORER = "nc_dac_ep"
SAFETY_GATE_EP_SCORER = "safety_gate_ep"
HYDRA_PRODUCT_SCORER = "hydra_product"
HYDRA_PRODUCT_DEFAULT_EP_EXPONENT = 3.0
# Mix λ in gtrs + λ log(TTC). Weak enough that typical TTC ranking
# matches nc_dac_ep; TTC≈0 still loses. Do not hard-mask EP.
HYDRA_EXTRA_HEAD_EXPONENT = 0.25
VALID_SCORER_MODES = frozenset(
    {
        RELEASE_SCORER,
        NC_DAC_EP_SCORER,
        SAFETY_GATE_EP_SCORER,
        HYDRA_PRODUCT_SCORER,
    }
)
NC_DAC_EP_BASED_SCORERS = frozenset(
    {NC_DAC_EP_SCORER, SAFETY_GATE_EP_SCORER, HYDRA_PRODUCT_SCORER}
)
VALID_SPEED_PROXIES = frozenset({"longitudinal", "longitudinal_0p5s", "path_length"})
_HYDRA_NC_KEYS = ("no_at_fault_collisions", "nc")
_HYDRA_DAC_KEYS = ("drivable_area_compliance", "dac")
_HYDRA_EP_KEYS = ("ego_progress", "ep")
# Hydra-MDP extras only. TTC is in every GTRS result; comfort usually is not.
_HYDRA_OPTIONAL_PRODUCT_HEADS = (
    ("time_to_collision_within_bound", "ttc"),
    ("comfort",),
)


def _speed_reranked_scores(
    base_scores: Tensor,
    vocabulary: Tensor,
    *,
    top_k: int,
    speed_weight: float,
    speed_proxy: str = "longitudinal",
    curvature_weight: float = 0.0,
    heading_change_weight: float = 0.0,
) -> Tensor:
    if speed_proxy not in VALID_SPEED_PROXIES:
        raise ValueError(
            f"speed_proxy must be one of {sorted(VALID_SPEED_PROXIES)}; "
            f"got {speed_proxy!r}"
        )
    if speed_weight == 0.0 and curvature_weight == 0.0 and heading_change_weight == 0.0:
        return base_scores
    if top_k <= 0 or top_k > base_scores.shape[1]:
        raise ValueError("top_k must be between 1 and the vocabulary size")
    if vocabulary.ndim != 3 or vocabulary.shape[0] != base_scores.shape[1]:
        raise ValueError("vocabulary must align with the candidate scores")
    for name, weight in (
        ("speed_weight", speed_weight),
        ("curvature_weight", curvature_weight),
        ("heading_change_weight", heading_change_weight),
    ):
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    top_scores, top_indices = base_scores.topk(top_k, dim=1)
    adjusted = top_scores

    def standardized(metric: Tensor) -> Tensor:
        top_metric = metric[top_indices]
        centered = top_metric - top_metric.mean(dim=1, keepdim=True)
        scale = centered.square().mean(dim=1, keepdim=True).sqrt()
        return centered / scale.clamp_min(torch.finfo(base_scores.dtype).eps)

    if speed_weight > 0.0:
        if speed_proxy == "longitudinal":
            speed_metric = (vocabulary[:, -1, 0] - vocabulary[:, 0, 0]).clamp_min(0.0)
        elif speed_proxy == "longitudinal_0p5s":
            # Vocabulary poses start at 0.1 s, so index 4 is the 0.5 s pose.
            speed_metric = vocabulary[:, 4, 0].clamp_min(0.0)
        else:
            speed_metric = torch.linalg.vector_norm(
                torch.diff(vocabulary[..., :2], dim=1), dim=-1
            ).sum(dim=1)
        adjusted = adjusted + speed_weight * standardized(speed_metric)
    if curvature_weight > 0.0 or heading_change_weight > 0.0:
        delta_yaw = torch.diff(vocabulary[..., 2], dim=1)
        delta_yaw = torch.atan2(delta_yaw.sin(), delta_yaw.cos()).abs()
        if heading_change_weight > 0.0:
            adjusted = adjusted - heading_change_weight * standardized(
                delta_yaw.mean(dim=1)
            )
        if curvature_weight > 0.0:
            segment_length = torch.linalg.vector_norm(
                torch.diff(vocabulary[..., :2], dim=1), dim=-1
            )
            max_curvature = (delta_yaw / segment_length.clamp_min(0.1)).amax(dim=1)
            adjusted = adjusted - curvature_weight * standardized(max_curvature)

    scores = torch.full_like(base_scores, -torch.inf)
    return scores.scatter(1, top_indices, adjusted)


def _first_score_tensor(
    result: Mapping[str, Tensor], names: tuple[str, ...]
) -> Tensor | None:
    for name in names:
        value = result.get(name)
        if value is not None:
            return value
    return None


def _hydra_product_scores(
    result: Mapping[str, Tensor], ep_exponent: float
) -> Tensor:
    if not math.isfinite(ep_exponent) or ep_exponent <= 0.0 or ep_exponent > 20.0:
        raise ValueError("ep_exponent must be finite and in (0, 20]")
    nc = _first_score_tensor(result, _HYDRA_NC_KEYS)
    dac = _first_score_tensor(result, _HYDRA_DAC_KEYS)
    ep = _first_score_tensor(result, _HYDRA_EP_KEYS)
    if nc is None or dac is None or ep is None:
        raise ValueError("hydra_product requires NC, DAC, and EP tensors in result")
    # Log-sum of NC×DAC×EP^α (same as nc_dac_ep). EP stays in the mix.
    score = (
        nc.sigmoid().log()
        + dac.sigmoid().log()
        + ep_exponent * ep.sigmoid().log()
    )
    for names in _HYDRA_OPTIONAL_PRODUCT_HEADS:
        extra = _first_score_tensor(result, names)
        if extra is not None:
            score = score + HYDRA_EXTRA_HEAD_EXPONENT * extra.sigmoid().log()
    return score


def _trajectory_scores(
    result: Mapping[str, Tensor],
    scorer_mode: str = RELEASE_SCORER,
    ep_exponent: float = 1.0,
) -> Tensor:
    if scorer_mode == HYDRA_PRODUCT_SCORER:
        return _hydra_product_scores(result, ep_exponent)
    if scorer_mode in {NC_DAC_EP_SCORER, SAFETY_GATE_EP_SCORER}:
        if not math.isfinite(ep_exponent) or ep_exponent <= 0.0 or ep_exponent > 20.0:
            raise ValueError("ep_exponent must be finite and in (0, 20]")
        return (
            result["no_at_fault_collisions"].sigmoid().log()
            + result["drivable_area_compliance"].sigmoid().log()
            + ep_exponent * result["ego_progress"].sigmoid().log()
        )
    if scorer_mode != RELEASE_SCORER:
        raise ValueError(
            f"scorer_mode must be one of {sorted(VALID_SCORER_MODES)}; "
            f"got {scorer_mode!r}"
        )
    return (
        0.03 * result["imi"].softmax(-1).log()
        + 0.1 * result["traffic_light_compliance"].sigmoid().log()
        + 0.1 * result["no_at_fault_collisions"].sigmoid().log()
        + 0.9 * result["drivable_area_compliance"].sigmoid().log()
        + 0.2 * result["driving_direction_compliance"].sigmoid().log()
        + 6.0
        * (
            7.0 * result["time_to_collision_within_bound"].sigmoid()
            + 7.0 * result["ego_progress"].sigmoid()
            + 3.0 * result["lane_keeping"].sigmoid()
        ).log()
    )


def _safety_gated_ep_scores(
    result: Mapping[str, Tensor], anchor_scores: Tensor
) -> Tensor:
    anchor_indices = anchor_scores.argmax(dim=1, keepdim=True)
    nc = result["no_at_fault_collisions"].sigmoid()
    dac = result["drivable_area_compliance"].sigmoid()
    eligible = (nc >= nc.gather(1, anchor_indices)) & (
        dac >= dac.gather(1, anchor_indices)
    )
    ep_scores = result["ego_progress"].sigmoid().log()
    return ep_scores.masked_fill(~eligible, -torch.inf)


class AgentHead(nn.Module):
    def __init__(self, num_agents: int, d_ffn: int, d_model: int) -> None:
        super().__init__()
        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn
        self._mlp_states = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, 5),
        )
        self._mlp_label = nn.Sequential(nn.Linear(d_model, 1))


class GTRSDenseModel(nn.Module):
    def __init__(
        self,
        config: GTRSDenseConfig,
        *,
        vocab: Tensor,
        scorer_mode: str = RELEASE_SCORER,
        ep_exponent: float = 1.0,
        speed_top_k: int = 0,
        speed_weight: float = 0.0,
        speed_proxy: str = "longitudinal",
        curvature_weight: float = 0.0,
        heading_change_weight: float = 0.0,
        spatial_anchors: Tensor | None = None,
        spatial_plan_topk: int = 32,
        spatial_rerank_score_margin: float = SCORE_MARGIN,
        spatial_route_arc_gate: bool = True,
        spatial_route_min_reference_dev: float = ROUTE_MIN_REFERENCE_DEV,
        spatial_route_min_improvement: float = ROUTE_MIN_IMPROVEMENT,
        spatial_route_arc_center_scale: float | None = None,
        spatial_reference_safety_gate: bool = False,
        keep_base_scores: bool = False,
        candidate_utility: bool = False,
        prefix_progress: bool = False,
    ) -> None:
        super().__init__()
        if (
            not math.isfinite(spatial_rerank_score_margin)
            or spatial_rerank_score_margin < 0.0
        ):
            raise ValueError(
                "spatial_rerank_score_margin must be finite and non-negative"
            )
        if (
            not math.isfinite(spatial_route_min_reference_dev)
            or spatial_route_min_reference_dev < 0.0
        ):
            raise ValueError(
                "spatial_route_min_reference_dev must be finite and non-negative"
            )
        if (
            not math.isfinite(spatial_route_min_improvement)
            or spatial_route_min_improvement < 0.0
        ):
            raise ValueError(
                "spatial_route_min_improvement must be finite and non-negative"
            )
        if spatial_route_arc_center_scale is not None and (
            not math.isfinite(spatial_route_arc_center_scale)
            or spatial_route_arc_center_scale <= 0.0
        ):
            raise ValueError("spatial_route_arc_center_scale must be finite and positive")
        self.spatial_rerank_score_margin = spatial_rerank_score_margin
        self.spatial_route_arc_gate = spatial_route_arc_gate
        self.spatial_route_min_reference_dev = spatial_route_min_reference_dev
        self.spatial_route_min_improvement = spatial_route_min_improvement
        self.spatial_route_arc_center_scale = spatial_route_arc_center_scale
        self.spatial_reference_safety_gate = spatial_reference_safety_gate
        self._da_progress_head: nn.Module | None = None
        self._switch_gate: nn.Module | None = None
        self._candidate_utility_head: CandidateUtilityHead | None = None
        self._prefix_progress_head: PrefixProgressHead | None = None
        if prefix_progress and not candidate_utility:
            raise ValueError("prefix_progress requires candidate_utility=True")
        if candidate_utility:
            if spatial_anchors is None or config.backbone_type != "vov":
                raise ValueError("candidate_utility requires shared VoV spatial path")
            self._candidate_utility_head = CandidateUtilityHead(
                query_dim=config.tf_d_model, hidden_dim=128
            )
            if prefix_progress:
                self._prefix_progress_head = PrefixProgressHead(
                    query_dim=config.tf_d_model, hidden_dim=128)
        if spatial_anchors is not None and config.backbone_type != "vov":
            raise ValueError(
                "the spatial path head reads the VoV front-camera grid; "
                "set backbone_type='vov'"
            )
        expected_vocab_shape = (config.vocab_size, config.num_poses, 3)
        if tuple(vocab.shape) != expected_vocab_shape:
            raise ValueError(
                f"vocabulary must have shape {expected_vocab_shape}; got {tuple(vocab.shape)}"
            )
        if not vocab.is_floating_point():
            raise ValueError("vocabulary must use a floating-point dtype")

        self._query_splits = [config.num_bounding_boxes]
        self._config = config
        if config.backbone_type == "resnet":
            self._backbone = TransfuserBackbone(config)
            self._bev_downscale = nn.Conv2d(512 + 64, config.tf_d_model, 1)
            keyval_tokens = 4096
        else:
            self._backbone = VoVBackbone(config)
            self.downscale_layer = nn.Conv2d(
                self._backbone.img_feat_c, config.tf_d_model, 1
            )
            keyval_tokens = config.img_vert_anchors * config.img_horz_anchors
        self._keyval_embedding = nn.Embedding(keyval_tokens, config.tf_d_model)
        self._status_encoding = nn.Linear(8 * config.num_ego_status, config.tf_d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self._tf_decoder = nn.TransformerDecoder(decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            config.num_bounding_boxes,
            config.tf_d_ffn,
            config.tf_d_model,
        )
        self._trajectory_head = GTRSDenseTrajectoryHead(
            num_poses=config.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            vocab=vocab,
            config=config,
            scorer_mode=scorer_mode,
            ep_exponent=ep_exponent,
            speed_top_k=speed_top_k,
            speed_weight=speed_weight,
            speed_proxy=speed_proxy,
            curvature_weight=curvature_weight,
            heading_change_weight=heading_change_weight,
            spatial_plan_topk=spatial_plan_topk if spatial_anchors is not None else 0,
            keep_base_scores=keep_base_scores or spatial_anchors is not None,
        )
        self._trajectory_head.candidate_utility = candidate_utility
        self._spatial_head = None
        if spatial_anchors is not None:
            self._spatial_head = SpatialPathHead(
                d_model=config.tf_d_model,
                d_ffn=config.tf_d_ffn,
                nhead=config.tf_num_head,
                anchors=spatial_anchors,
                in_channels=config.tf_d_model,
            )

    def _image_features(self, camera_feature: Tensor) -> Tensor:
        return self._image_features_and_front(camera_feature)[0]

    def _image_features_and_front(
        self, camera_feature: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if self._config.backbone_type == "vov":
            image_features = self._backbone(camera_feature)
            grid = self.downscale_layer(image_features)
            front = None
            if self._spatial_head is not None:
                start, stop = front_view_columns(grid.shape[-1])
                front = grid[..., start:stop].contiguous()
            tokens = grid.flatten(-2, -1).permute(0, 2, 1).contiguous()
            return tokens, front, grid
        upscaled, bev_feature, _ = self._backbone(camera_feature)
        if upscaled is None:
            raise RuntimeError("GTRS-Dense requires the backbone top-down features")
        bev_feature = F.interpolate(
            bev_feature,
            size=upscaled.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        bev_feature = torch.cat((bev_feature, upscaled), dim=1)
        tokens = (
            self._bev_downscale(bev_feature)
            .flatten(-2, -1)
            .permute(0, 2, 1)
            .contiguous()
        )
        return tokens, None, None

    def forward(
        self, features: Mapping[str, Tensor | list[Tensor]]
    ) -> dict[str, Tensor]:
        status_feature = features["status_feature"]
        if isinstance(status_feature, list):
            status_feature = status_feature[-1]
        camera_feature = features["camera_feature"]
        if isinstance(camera_feature, list):
            camera_feature = camera_feature[-1]

        status_encoding = self._status_encoding(status_feature)
        keyval, front, grid = self._image_features_and_front(camera_feature)
        keyval = keyval + self._keyval_embedding.weight[None]
        result = self._trajectory_head(keyval, status_encoding)
        plan_query = result.pop("spatial_plan_query", None)
        candidate_features = result.pop("candidate_utility_features", None)
        if self._spatial_head is not None:
            command = status_feature[:, :4].argmax(dim=-1)
            result.update(self._spatial_head(front, plan_query, command))
            if self._da_progress_head is not None:
                if grid is None:
                    raise RuntimeError("DA progress head requires the VoV feature grid")
                path = result["spatial_path"]
                path_mask = torch.ones(path.shape[:2], device=path.device, dtype=torch.bool)
                result.update(self._da_progress_head(grid, path, path_mask, status_feature))
                valid = torch.isfinite(result["progress_logits"]).all(dim=-1)
                valid = valid & torch.isfinite(path).flatten(1).all(dim=1)
                result["progress_valid"] = valid
                result["progress_s4"] = result["progress_s4"].masked_fill(~valid, float("nan"))
            if self._candidate_utility_head is not None:
                if candidate_features is None:
                    raise RuntimeError("candidate utility features are unavailable")
                self._rerank_utility(result, candidate_features, status_feature[:, 4:6])
            else:
                self._rerank_spatial(result)
        return result

    def _rerank_utility(self, result: dict[str, Tensor], candidate_features: Tensor,
                        ego_velocity: Tensor) -> None:
        if self._candidate_utility_head is None:
            raise RuntimeError("candidate utility head is not loaded")
        vocab = result["trajectory_vocab"]
        path = result["spatial_path"]
        if candidate_features.shape != (result["scores"].shape[0], vocab.shape[0], 256):
            raise ValueError("candidate utility features do not align with the full vocabulary")
        if not torch.isfinite(candidate_features).all() or not torch.isfinite(path).all():
            raise ValueError("candidate utility requires finite VoV features and predicted path")
        base_logits = torch.stack(
            (result["no_at_fault_collisions"], result["drivable_area_compliance"],
             result["ego_progress"]), dim=-1
        )
        if not torch.isfinite(base_logits).all():
            raise ValueError("candidate utility base logits contain non-finite values")
        geometry = torch.cat([
            candidate_geometry(
                vocab[start:start + 1024].unsqueeze(0).expand(candidate_features.shape[0], -1, -1, -1),
                path,
            )
            for start in range(0, vocab.shape[0], 1024)
        ], dim=1)
        utility = self._candidate_utility_head(candidate_features, base_logits, geometry)
        scores = utility["candidate_utility_log_scores"]
        if self._prefix_progress_head is not None:
            arcs = prefix_candidate_arcs(vocab[None].expand(candidate_features.shape[0], -1, -1, -1))
            prefix = self._prefix_progress_head(candidate_features, geometry, arcs, ego_velocity)
            scores = prefix_utility_log_score(utility["candidate_utility_logits"],
                                             prefix["prefix_progress_log_score"])
            result.update(prefix)
            result["prefix_candidate_arcs"] = arcs
            result["prefix_ego_velocity"] = ego_velocity
            result["prefix_utility_log_scores"] = scores
            result["prefix_base_vocab_ids"] = utility["candidate_utility_log_scores"].argmax(dim=1)
            result["prefix_selected_vocab_ids"] = scores.argmax(dim=1)
        if not torch.isfinite(scores).all():
            raise ValueError("candidate utility scores contain non-finite values")
        index = scores.argmax(dim=1)
        result["trajectory_scored"] = result["trajectory"]
        result["selected_indices_scored"] = result["selected_indices"]
        result["candidate_utility_features"] = candidate_features
        result["candidate_utility_base_logits"] = base_logits
        result["candidate_utility_geometry"] = geometry
        result["candidate_utility_vocab_ids"] = torch.arange(vocab.shape[0], device=vocab.device)
        result.update(utility)
        result["selected_indices"] = index
        result["trajectory"] = vocab[index]

    def _rerank_spatial(self, result: dict[str, Tensor]) -> None:
        """Route-constrained pick over the whole vocabulary (spatial_path.route_constrained_pick).

        The pool is read from base_scores, before the speed top-k cut, so rows the
        cut dropped still compete: within spatial_rerank_score_margin of the best base
        score, sigmoid(NC) and sigmoid(DAC) >= ROUTE_MIN_SAFETY_PROB, and longitudinally
        comparable to the scored pick unless spatial_route_arc_gate is False. The one
        with the smallest max deviation from the path is executed as the
        vocabulary row as is; otherwise, or when the scored pick's own max deviation is
        below spatial_route_min_reference_dev, or that row is not at least
        spatial_route_min_improvement metres closer than the scored pick, the scored pick
        is kept. The scored pick stays in trajectory_scored / selected_indices_scored.

        With spatial_route_arc_center_scale set, the arc gate is centred on that
        fraction of the scored pick's arc instead of the arc itself, so rows that
        drive less far than an over-fast scored pick can compete; None keeps the
        scored pick's arc.
        """
        vocab = result["trajectory_vocab"]
        progress = result.get("progress_s4")
        if progress is None and self.spatial_route_arc_center_scale is not None:
            xy = vocab[result["selected_indices"]][..., :2]
            xy = torch.cat([xy.new_zeros(xy.shape[0], 1, 2), xy], dim=1)
            arc = torch.linalg.vector_norm(xy[:, 1:] - xy[:, :-1], dim=-1).sum(dim=-1)
            progress = self.spatial_route_arc_center_scale * arc
        collision_prob = result["no_at_fault_collisions"].sigmoid()
        drivable_prob = result["drivable_area_compliance"].sigmoid()
        path = result["spatial_path"]
        path_valid = torch.isfinite(path).flatten(1).all(dim=1)
        path = torch.where(path_valid[:, None, None], path, torch.zeros_like(path))
        if getattr(self, "spatial_reference_safety_gate", False):
            reference = result["selected_indices"][:, None]
            collision_prob = collision_prob.masked_fill(
                collision_prob < collision_prob.gather(1, reference), 0.0
            )
            drivable_prob = drivable_prob.masked_fill(
                drivable_prob < drivable_prob.gather(1, reference), 0.0
            )
        index, deviation, pool_size = route_constrained_pick(
            result["base_scores"],
            collision_prob,
            drivable_prob,
            vocab,
            path,
            reference=result["selected_indices"],
            score_margin=self.spatial_rerank_score_margin,
            arc_gate=self.spatial_route_arc_gate,
            progress=progress,
            min_reference_deviation=self.spatial_route_min_reference_dev,
            min_improvement=self.spatial_route_min_improvement,
        )
        c0_index = index
        if self._switch_gate is not None:
            reference = result["selected_indices"]
            gate_features = switch_features(result, reference, c0_index, deviation, path)
            gate_probability = self._switch_gate(gate_features).sigmoid()
            index = torch.where(gate_probability >= GATE_THRESHOLD, c0_index, reference)
            result["selected_indices_c0"] = c0_index
            result["trajectory_c0"] = vocab[c0_index]
            result["switch_gate_probability"] = gate_probability
        result["trajectory_scored"] = result["trajectory"]
        result["selected_indices_scored"] = result["selected_indices"]
        result["spatial_rerank_deviation"] = deviation.lateral.gather(1, index[:, None])[:, 0]
        result["spatial_coverage_shortfall"] = deviation.shortfall.gather(
            1, index[:, None]
        )[:, 0]
        result["spatial_route_pool_size"] = pool_size
        result["selected_indices"] = index
        result["trajectory"] = vocab[index]


class GTRSDenseTrajectoryHead(nn.Module):
    def __init__(
        self,
        *,
        num_poses: int,
        d_ffn: int,
        d_model: int,
        nhead: int,
        nlayers: int,
        vocab: Tensor,
        config: GTRSDenseConfig,
        scorer_mode: str = RELEASE_SCORER,
        ep_exponent: float = 1.0,
        speed_top_k: int = 0,
        speed_weight: float = 0.0,
        speed_proxy: str = "longitudinal",
        curvature_weight: float = 0.0,
        heading_change_weight: float = 0.0,
        spatial_plan_topk: int = 0,
        keep_base_scores: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(spatial_plan_topk, int) or spatial_plan_topk < 0:
            raise ValueError("spatial_plan_topk must be a non-negative integer")
        self.spatial_plan_topk = spatial_plan_topk
        # base_scores: every row's score before the speed top-k cut, for the spatial pick.
        self.keep_base_scores = keep_base_scores
        if scorer_mode not in VALID_SCORER_MODES:
            raise ValueError(
                f"scorer_mode must be one of {sorted(VALID_SCORER_MODES)}; "
                f"got {scorer_mode!r}"
            )
        self.config = config
        self.scorer_mode = scorer_mode
        self.ep_exponent = ep_exponent
        if not isinstance(speed_top_k, int) or speed_top_k < 0:
            raise ValueError("speed_top_k must be a non-negative integer")
        if speed_top_k > vocab.shape[0]:
            raise ValueError("speed_top_k cannot exceed the vocabulary size")
        if not math.isfinite(speed_weight) or speed_weight < 0.0:
            raise ValueError("speed_weight must be finite and non-negative")
        if speed_weight > 0.0 and speed_top_k == 0:
            raise ValueError(
                "speed_top_k must be positive when speed_weight is enabled"
            )
        nc_dac_ep_scorers = NC_DAC_EP_BASED_SCORERS
        if speed_weight > 0.0 and scorer_mode not in nc_dac_ep_scorers:
            raise ValueError("speed reranking requires an NC/DAC/EP-based scorer")
        if speed_proxy not in VALID_SPEED_PROXIES:
            raise ValueError(
                f"speed_proxy must be one of {sorted(VALID_SPEED_PROXIES)}; "
                f"got {speed_proxy!r}"
            )
        for name, weight in (
            ("curvature_weight", curvature_weight),
            ("heading_change_weight", heading_change_weight),
        ):
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (curvature_weight > 0.0 or heading_change_weight > 0.0) and speed_top_k == 0:
            raise ValueError("trackability reranking requires a positive speed_top_k")
        if (
            curvature_weight > 0.0 or heading_change_weight > 0.0
        ) and scorer_mode not in nc_dac_ep_scorers:
            raise ValueError(
                "trackability reranking requires an NC/DAC/EP-based scorer"
            )
        self.speed_top_k = speed_top_k
        self.speed_weight = speed_weight
        self.speed_proxy = speed_proxy
        self.curvature_weight = curvature_weight
        self.heading_change_weight = heading_change_weight
        self._num_poses = num_poses
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model,
                nhead,
                d_ffn,
                dropout=0.0,
                batch_first=True,
            ),
            nlayers,
        )
        self.vocab = nn.Parameter(vocab.detach().clone(), requires_grad=False)
        self.heads = nn.ModuleDict(
            {
                "no_at_fault_collisions": _score_head(d_model, d_ffn),
                "drivable_area_compliance": _score_head(d_model, d_ffn),
                "time_to_collision_within_bound": _score_head(d_model, d_ffn),
                "ego_progress": _score_head(d_model, d_ffn),
                "driving_direction_compliance": _score_head(d_model, d_ffn),
                "lane_keeping": _score_head(d_model, d_ffn),
                "traffic_light_compliance": _score_head(d_model, d_ffn),
                "imi": nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, d_ffn),
                    nn.ReLU(),
                    nn.Linear(d_ffn, 1),
                ),
            }
        )
        self.normalize_vocab_pos = config.normalize_vocab_pos
        if self.normalize_vocab_pos:
            self.encoder = MemoryEffTransformer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.0,
            )
        self.use_nerf = config.use_nerf
        self.pos_embed = nn.Sequential(
            nn.Linear(num_poses * 3, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        self._inference_vocab_cache = InferenceTensorCache()

    def _embedded_vocabulary(self, batch_size: int) -> Tensor:
        def compute() -> Tensor:
            embedded = self.pos_embed(self.vocab.detach().reshape(self.vocab.shape[0], -1))[None]
            if self.normalize_vocab_pos:
                embedded = self.encoder(embedded)
            # Preserve the original contiguous layout and attention arithmetic.
            return embedded.repeat(batch_size, 1, 1)

        tensors = [self.vocab, *self.pos_embed.parameters()]
        if self.normalize_vocab_pos:
            tensors.extend(self.encoder.parameters())
        return self._inference_vocab_cache.get(self, tensors, batch_size, compute)

    def forward(
        self, bev_feature: Tensor, status_encoding: Tensor
    ) -> dict[str, Tensor]:
        vocab = self.vocab.detach()
        vocab_count = vocab.shape[0]
        embedded_vocab = self._embedded_vocabulary(bev_feature.shape[0])
        trajectory_features = self.transformer(embedded_vocab, bev_feature)
        score_features = trajectory_features + status_encoding.unsqueeze(1)

        result = {
            name: head(score_features).squeeze(-1) for name, head in self.heads.items()
        }
        if getattr(self, "candidate_utility", False):
            result["candidate_utility_features"] = score_features
        base_scores = _trajectory_scores(result, self.scorer_mode, self.ep_exponent)
        scores = _speed_reranked_scores(
            base_scores,
            vocab,
            top_k=self.speed_top_k,
            speed_weight=self.speed_weight,
            speed_proxy=self.speed_proxy,
            curvature_weight=self.curvature_weight,
            heading_change_weight=self.heading_change_weight,
        )
        if self.scorer_mode == SAFETY_GATE_EP_SCORER:
            scores = _safety_gated_ep_scores(result, scores)
        selected_indices = scores.argmax(1)
        result.update(
            {
                "dropout_indices": torch.arange(vocab_count, device=vocab.device),
                "trajectory_vocab_dropout": vocab,
                "trajectory": vocab[selected_indices],
                "trajectory_vocab": vocab,
                "selected_indices": selected_indices,
                "scores": scores,
            }
        )
        if self.keep_base_scores:
            result["base_scores"] = base_scores
        if self.spatial_plan_topk > 0:
            # Head context only. Ranked by the release weights, as in GTRS
            # HydraTrajHead.forward where the spatial head was trained.
            plan_scores = _trajectory_scores(result, RELEASE_SCORER)
            top = plan_scores.topk(
                min(self.spatial_plan_topk, vocab_count), dim=1
            ).indices
            result["spatial_plan_query"] = score_features.gather(
                1, top[..., None].expand(-1, -1, score_features.shape[-1])
            )
        return result


def _score_head(d_model: int, d_ffn: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_ffn),
        nn.ReLU(),
        nn.Linear(d_ffn, 1),
    )
