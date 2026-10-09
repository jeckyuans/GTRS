# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .preprocessing import CAMERA_IDS, build_status_feature, preprocess_images
from .simscale_gtrs_dense.config import GTRSDenseConfig
from .simscale_gtrs_dense.candidate_utility import CandidateUtilityHead
from .simscale_gtrs_dense.prefix_progress import PrefixProgressHead
from .simscale_gtrs_dense.da_progress import SharedVoVDAProgressHead
from .simscale_gtrs_dense.model import RELEASE_SCORER, GTRSDenseModel
from .simscale_gtrs_dense.spatial_path import (
    ROUTE_MIN_IMPROVEMENT,
    ROUTE_MIN_REFERENCE_DEV,
    load_clustered_anchor_offsets,
    walk_path_at_reference_arc,
)
from .simscale_gtrs_dense.switch_gate import FEATURE_NAMES, GATE_THRESHOLD, SwitchGate

LOGGER = logging.getLogger(__name__)

CHECKPOINT_PREFIX = "agent.model."
VOCABULARY_KEY = f"{CHECKPOINT_PREFIX}_trajectory_head.vocab"
SPATIAL_ANCHORS_KEY = "_spatial_head.anchors"
SWITCH_GATE_PREFIX = "_switch_gate."
UTILITY_PREFIX = "_candidate_utility_head."
PREFIX_PROGRESS_PREFIX = "_prefix_progress_head."
SPATIAL_OUTPUT_KEYS = (
    "spatial_cls",
    "spatial_offsets",
    "spatial_positions",
    "spatial_mode",
    "spatial_path",
    "spatial_rerank_deviation",
    "spatial_coverage_shortfall",
    "spatial_route_pool_size",
    "selected_indices_scored",
    "trajectory_scored",
    "trajectory_route_pick",
    "selected_indices_c0",
    "trajectory_c0",
    "switch_gate_probability",
    "progress_s4",
    "progress_valid",
)
# What a separate path model contributes; the pick itself runs on the scoring model's heads.
SPATIAL_HEAD_OUTPUT_KEYS = (
    "spatial_cls",
    "spatial_offsets",
    "spatial_positions",
    "spatial_mode",
    "spatial_path",
)
NAVHARD_VOCABULARY_SHAPE = (8192, 40, 3)
CHECKPOINT_VOCABULARY_SHAPE = (16384, 40, 3)
OFFICIAL_VOCABULARIES = {
    3_932_288: (
        "cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35",
        NAVHARD_VOCABULARY_SHAPE,
    ),
    7_864_448: (
        "e8c29cfc25add59ae8b64769a4554c6518878726178c0bd889fc8518ebe1261d",
        CHECKPOINT_VOCABULARY_SHAPE,
    ),
}


@dataclass(frozen=True)
class InferenceInput:
    images: Mapping[str, np.ndarray]
    command_one_hot: np.ndarray
    velocity_xy: np.ndarray
    acceleration_xy: np.ndarray

    def __post_init__(self) -> None:
        missing = [
            camera_id for camera_id in CAMERA_IDS if camera_id not in self.images
        ]
        if missing:
            raise ValueError(f"missing required cameras: {', '.join(missing)}")


@dataclass(frozen=True)
class Prediction:
    # With GTRS_SPATIAL_PATH=1 this is the route-constrained pick, an unmodified
    # vocabulary row; otherwise the score argmax. With GTRS_SPATIAL_EXECUTE_PATH=1 as
    # well it is the spatial path walked at the scored pick's arc lengths instead.
    # It is what the driver turns into the plan.
    trajectory: np.ndarray
    # spatial_* outputs plus the pre-rerank trajectory_scored, for logs only.
    spatial: dict[str, np.ndarray] = field(default_factory=dict)
    trace: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SwitchChoice:
    trajectory: np.ndarray
    selected_index: int
    positive_streak: int
    confirmed: bool
    reason: str


def temporal_switch_choice(
    positive_streak: int, confirmed: bool, prediction: Prediction
) -> SwitchChoice:
    """Choose only the original W0 or C0 row using one session's prior state."""
    spatial = prediction.spatial
    required = ("trajectory_scored", "trajectory_c0", "selected_indices_scored",
                "selected_indices_c0", "switch_gate_probability")
    if any(key not in spatial for key in required):
        raise ValueError("C4 temporal confirmation requires a merged C3 switch gate")
    reference = int(spatial["selected_indices_scored"])
    c0 = int(spatial["selected_indices_c0"])
    probability = float(spatial["switch_gate_probability"])
    positive = math.isfinite(probability) and probability >= 0.5 and c0 != reference
    next_streak = positive_streak + 1 if positive else 0
    next_confirmed = confirmed or next_streak >= 2
    trajectory = np.asarray(
        spatial["trajectory_c0"] if next_confirmed else spatial["trajectory_scored"]
    )
    if trajectory.shape != (40, 3) or not np.isfinite(trajectory).all():
        raise ValueError("C4 selected vocabulary row must be a finite (40,3) trajectory")
    reason = "latched_c0" if confirmed else (
        "confirmed_c0" if next_confirmed else "awaiting_second_positive" if positive else "w0"
    )
    return SwitchChoice(trajectory.copy(), c0 if next_confirmed else reference,
                        next_streak, next_confirmed, reason)


def _validate_vocabulary(
    vocabulary: Tensor,
    *,
    expected_shape: tuple[int, int, int],
    source: str,
) -> None:
    if tuple(vocabulary.shape) != expected_shape:
        raise ValueError(
            f"{source} vocabulary must have shape {expected_shape}; "
            f"got {tuple(vocabulary.shape)}"
        )
    if vocabulary.dtype != torch.float32:
        raise ValueError(f"{source} vocabulary must use dtype float32")
    if not torch.isfinite(vocabulary).all():
        raise ValueError(f"{source} vocabulary contains non-finite values")


def load_navhard_vocabulary(path: str | PathLike[str]) -> Tensor:
    vocabulary_path = Path(path)
    try:
        size = vocabulary_path.stat().st_size
    except OSError as exc:
        raise ValueError(
            f"NAVHARD vocabulary is unavailable: {vocabulary_path}"
        ) from exc
    vocabulary_spec = OFFICIAL_VOCABULARIES.get(size)
    if vocabulary_spec is None:
        raise ValueError(
            "official vocabulary size "
            f"{size}, expected one of {sorted(OFFICIAL_VOCABULARIES)}"
        )
    expected_sha, expected_shape = vocabulary_spec

    digest = hashlib.sha256()
    with vocabulary_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha = digest.hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(
            "official vocabulary SHA256 " f"{actual_sha}, expected {expected_sha}"
        )

    try:
        array = np.load(vocabulary_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("NAVHARD vocabulary is not a valid NPY array") from exc
    vocabulary = torch.from_numpy(np.asarray(array).copy())
    _validate_vocabulary(
        vocabulary,
        expected_shape=expected_shape,
        source="official",
    )
    return vocabulary


def normalized_checkpoint(
    payload: object,
    inference_vocabulary: Tensor,
    *,
    allow_gate_metadata: bool = False,
    allow_utility_metadata: bool = False,
    allow_prefix_metadata: bool = False,
) -> dict[str, Tensor]:
    expected_keys = {"state_dict"}
    if allow_gate_metadata:
        expected_keys.add("switch_gate_metadata")
    if allow_utility_metadata:
        expected_keys.add("candidate_utility_metadata")
    if allow_prefix_metadata:
        expected_keys.add("prefix_progress_metadata")
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(
            f"checkpoint payload must contain exactly {sorted(expected_keys)}"
        )
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint state_dict must be a dict")

    normalized: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not key.startswith(CHECKPOINT_PREFIX):
            raise ValueError(f"unexpected checkpoint key: {key}")
        if not isinstance(value, Tensor):
            raise ValueError(f"checkpoint value for {key} must be a tensor")
        normalized[key[len(CHECKPOINT_PREFIX):]] = value

    checkpoint_vocabulary = state_dict.get(VOCABULARY_KEY)
    if not isinstance(checkpoint_vocabulary, Tensor):
        raise ValueError("checkpoint vocabulary tensor is missing")
    _validate_vocabulary(
        checkpoint_vocabulary,
        expected_shape=CHECKPOINT_VOCABULARY_SHAPE,
        source="checkpoint",
    )
    inference_shape = tuple(inference_vocabulary.shape)
    supported_shapes = {spec[1] for spec in OFFICIAL_VOCABULARIES.values()}
    if inference_shape not in supported_shapes:
        raise ValueError(
            "inference vocabulary must have a supported shape; "
            f"got {inference_shape}"
        )
    _validate_vocabulary(
        inference_vocabulary,
        expected_shape=inference_shape,
        source="inference",
    )
    normalized["_trajectory_head.vocab"] = inference_vocabulary
    return normalized


def _spatial_anchors(
    state_dict: dict[str, Tensor],
    spatial_anchor_path: str | PathLike[str] | None,
) -> Tensor:
    # Unused by this model; present in checkpoints saved by GTRS HydraModel.
    state_dict.pop("_query_embedding.weight", None)
    spatial_anchors = state_dict.get(SPATIAL_ANCHORS_KEY)
    if spatial_anchors is not None:
        return spatial_anchors
    if not spatial_anchor_path:
        raise ValueError(
            "spatial_path needs _spatial_head.anchors in the "
            "checkpoint or spatial_anchor_path"
        )
    return load_clustered_anchor_offsets(spatial_anchor_path)


class GTRSDensePolicy:
    path_model: GTRSDenseModel | None = None
    spatial_execute_path: bool = False

    def __init__(
        self,
        checkpoint_path: str | PathLike[str],
        vocabulary_path: str | PathLike[str],
        device: str | torch.device = "cuda",
        scorer_mode: str = RELEASE_SCORER,
        ep_exponent: float = 1.0,
        speed_top_k: int = 0,
        speed_weight: float = 0.0,
        speed_proxy: str = "longitudinal",
        curvature_weight: float = 0.0,
        heading_change_weight: float = 0.0,
        warm_up: bool = True,
        backbone_type: str = "resnet",
        spatial_path: bool = False,
        spatial_anchor_path: str | PathLike[str] | None = None,
        spatial_plan_topk: int = 32,
        spatial_rerank_score_margin: float = 1.0,
        spatial_route_arc_gate: bool = True,
        spatial_path_checkpoint: str | PathLike[str] | None = None,
        spatial_route_min_reference_dev: float = ROUTE_MIN_REFERENCE_DEV,
        spatial_route_min_improvement: float = ROUTE_MIN_IMPROVEMENT,
        spatial_route_arc_center_scale: float | None = None,
        spatial_execute_path: bool = False,
        da_progress_checkpoint: str | PathLike[str] | None = None,
        spatial_reference_safety_gate: bool = False,
        candidate_utility: bool = False,
        prefix_progress: bool = False,
        prefix_base_sha256: str | None = None,
    ) -> None:
        if spatial_path_checkpoint is not None and not spatial_path:
            raise ValueError("spatial_path_checkpoint requires spatial_path")
        if spatial_execute_path and not spatial_path:
            raise ValueError("spatial_execute_path requires spatial_path")
        if da_progress_checkpoint is not None and (not spatial_path or backbone_type != "vov"):
            raise ValueError("da_progress_checkpoint requires spatial_path and VoV")
        if da_progress_checkpoint is not None and spatial_execute_path:
            raise ValueError("DA progress requires execution of a vocabulary row")
        if candidate_utility and (not spatial_path or backbone_type != "vov" or spatial_path_checkpoint is not None or spatial_execute_path):
            raise ValueError("candidate utility requires one merged VoV scorer and spatial path")
        if prefix_progress and not candidate_utility:
            raise ValueError("prefix progress requires candidate utility on the same VoV package")
        if prefix_progress and (not isinstance(prefix_base_sha256, str)
                                or len(prefix_base_sha256) != 64
                                or any(ch not in "0123456789abcdef" for ch in prefix_base_sha256)):
            raise ValueError("prefix progress requires explicit frozen base SHA256")
        self.spatial_execute_path = spatial_execute_path
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {self.device} was requested, but CUDA is not available"
            )
        self.dtype = torch.float32

        vocabulary = load_navhard_vocabulary(vocabulary_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        merged_gate = isinstance(payload, dict) and "switch_gate_metadata" in payload
        merged_utility = isinstance(payload, dict) and "candidate_utility_metadata" in payload
        merged_prefix = isinstance(payload, dict) and "prefix_progress_metadata" in payload
        if candidate_utility and not merged_utility:
            raise ValueError("candidate utility requires merged checkpoint metadata")
        if prefix_progress and not merged_prefix:
            raise ValueError("prefix progress requires merged checkpoint metadata")
        if merged_gate and (not spatial_path or backbone_type != "vov" or spatial_path_checkpoint is not None):
            raise ValueError("merged switch gate requires one VoV scorer with spatial path")
        state_dict = normalized_checkpoint(
            payload, vocabulary, allow_gate_metadata=merged_gate,
            allow_utility_metadata=merged_utility, allow_prefix_metadata=merged_prefix,
        )
        gate_state: dict[str, Tensor] = {}
        if merged_gate:
            metadata = payload["switch_gate_metadata"]
            if (
                not isinstance(metadata, dict)
                or metadata.get("feature_names") != list(FEATURE_NAMES)
                or metadata.get("threshold") != str(GATE_THRESHOLD)
                or metadata.get("label_convention") != "1_choose_c0"
            ):
                raise ValueError("switch gate metadata does not match deployment contract")
            gate_state = {
                key[len(SWITCH_GATE_PREFIX):]: state_dict.pop(key)
                for key in list(state_dict)
                if key.startswith(SWITCH_GATE_PREFIX)
            }
            if not gate_state:
                raise ValueError("merged checkpoint contains no switch gate tensors")
        utility_state: dict[str, Tensor] = {}
        if merged_utility:
            metadata = payload["candidate_utility_metadata"]
            if not isinstance(metadata, dict) or metadata.get("query_dim") != 256 or metadata.get("hidden_dim") != 128 or metadata.get("output_order") != ["route", "nc", "dac", "progress"]:
                raise ValueError("candidate utility metadata does not match deployment contract")
            utility_state = {
                key[len(UTILITY_PREFIX):]: state_dict.pop(key)
                for key in list(state_dict) if key.startswith(UTILITY_PREFIX)
            }
            if not utility_state:
                raise ValueError("merged checkpoint contains no candidate utility tensors")
        prefix_state: dict[str, Tensor] = {}
        if merged_prefix:
            metadata = payload["prefix_progress_metadata"]
            if (not isinstance(metadata, dict)
                    or set(metadata) != {"query_dim", "hidden_dim", "horizons", "base_model_sha256"}
                    or metadata["query_dim"] != 256 or metadata["hidden_dim"] != 128
                    or metadata["horizons"] != [0.5, 1.0, 2.0, 4.0]
                    or not isinstance(metadata["base_model_sha256"], str)
                    or len(metadata["base_model_sha256"]) != 64
                    or any(ch not in "0123456789abcdef" for ch in metadata["base_model_sha256"])
                    or (prefix_progress and metadata["base_model_sha256"] != prefix_base_sha256)):
                raise ValueError("prefix progress metadata does not match frozen U1 contract")
            prefix_state = {
                key[len(PREFIX_PROGRESS_PREFIX):]: state_dict.pop(key)
                for key in list(state_dict) if key.startswith(PREFIX_PROGRESS_PREFIX)
            }
            if not prefix_state:
                raise ValueError("merged checkpoint contains no prefix progress tensors")
        spatial_kwargs: dict[str, object] = {}
        # Keep the released scorer and backbone. The path checkpoint contributes
        # only its spatial head, which reads the scoring model's VoV feature grid.
        if spatial_path_checkpoint is not None:
            if backbone_type != "vov":
                raise ValueError("spatial_path_checkpoint requires backbone_type='vov'")
            path_state = normalized_checkpoint(
                torch.load(
                    spatial_path_checkpoint, map_location="cpu", weights_only=True
                ),
                vocabulary,
            )
            anchors = _spatial_anchors(path_state, spatial_anchor_path)
            path_head_state = {
                key[len("_spatial_head."):]: value
                for key, value in path_state.items()
                if key.startswith("_spatial_head.")
            }
            if not path_head_state:
                raise ValueError("spatial path checkpoint contains no _spatial_head tensors")
            spatial_kwargs = {
                "spatial_anchors": anchors,
                "spatial_plan_topk": spatial_plan_topk,
                "spatial_rerank_score_margin": spatial_rerank_score_margin,
                "spatial_route_arc_gate": spatial_route_arc_gate,
                "spatial_route_min_reference_dev": spatial_route_min_reference_dev,
                "spatial_route_min_improvement": spatial_route_min_improvement,
                "spatial_route_arc_center_scale": spatial_route_arc_center_scale,
                "spatial_reference_safety_gate": spatial_reference_safety_gate,
            }
            LOGGER.info(
                "Loading %d spatial-head tensors from %s on released scorer",
                len(path_head_state),
                spatial_path_checkpoint,
            )
        elif spatial_path:
            spatial_kwargs = {
                "spatial_anchors": _spatial_anchors(state_dict, spatial_anchor_path),
                "spatial_plan_topk": spatial_plan_topk,
                "spatial_rerank_score_margin": spatial_rerank_score_margin,
                "spatial_route_arc_gate": spatial_route_arc_gate,
                "spatial_route_min_reference_dev": spatial_route_min_reference_dev,
                "spatial_route_min_improvement": spatial_route_min_improvement,
                "spatial_route_arc_center_scale": spatial_route_arc_center_scale,
                "spatial_reference_safety_gate": spatial_reference_safety_gate,
            }
        config = GTRSDenseConfig(
            vocab_size=int(vocabulary.shape[0]), backbone_type=backbone_type
        )
        self.model = GTRSDenseModel(
            config,
            vocab=vocabulary,
            scorer_mode=scorer_mode,
            ep_exponent=ep_exponent,
            speed_top_k=speed_top_k,
            speed_weight=speed_weight,
            speed_proxy=speed_proxy,
            curvature_weight=curvature_weight,
            heading_change_weight=heading_change_weight,
            candidate_utility=candidate_utility,
            prefix_progress=prefix_progress,
            **spatial_kwargs,
        )
        if candidate_utility:
            assert self.model._candidate_utility_head is not None
            self.model._candidate_utility_head.load_state_dict(utility_state, strict=True)
        if prefix_progress:
            assert self.model._prefix_progress_head is not None
            self.model._prefix_progress_head.load_state_dict(prefix_state, strict=True)
        if spatial_path_checkpoint is not None:
            scorer_missing = self.model.load_state_dict(state_dict, strict=False)
            expected_missing = {
                key for key in self.model.state_dict() if key.startswith("_spatial_head.")
            }
            if set(scorer_missing.missing_keys) != expected_missing or scorer_missing.unexpected_keys:
                raise ValueError(
                    "released scorer checkpoint mismatch: "
                    f"missing={scorer_missing.missing_keys}, "
                    f"unexpected={scorer_missing.unexpected_keys}"
                )
            assert self.model._spatial_head is not None
            self.model._spatial_head.load_state_dict(path_head_state, strict=True)
        else:
            if candidate_utility:
                loaded = self.model.load_state_dict(state_dict, strict=False)
                expected_missing = {key for key in self.model.state_dict() if key.startswith(UTILITY_PREFIX)}
                if prefix_progress:
                    expected_missing |= {key for key in self.model.state_dict()
                                         if key.startswith(PREFIX_PROGRESS_PREFIX)}
                if set(loaded.missing_keys) != expected_missing or loaded.unexpected_keys:
                    raise ValueError(
                        "base checkpoint mismatch for candidate utility: "
                        f"missing={loaded.missing_keys}, unexpected={loaded.unexpected_keys}"
                    )
            else:
                self.model.load_state_dict(state_dict, strict=True)
        if merged_gate:
            gate = SwitchGate(gate_state["feature_mean"], gate_state["feature_std"])
            gate.load_state_dict(gate_state, strict=True)
            self.model._switch_gate = gate
            LOGGER.info("Loaded %d switch gate tensors from merged checkpoint %s", len(gate_state), checkpoint_path)
        if da_progress_checkpoint is not None:
            da_payload = torch.load(da_progress_checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(da_payload, dict) or "da_head_state_dict" not in da_payload:
                raise ValueError("DA checkpoint must contain da_head_state_dict")
            expected_config = {"d_model": 256, "nhead": 8, "n_points": 4, "n_rounds": 2}
            if da_payload.get("da_head_config") != expected_config:
                raise ValueError("DA checkpoint architecture does not match deployment head")
            da_head = SharedVoVDAProgressHead(**expected_config)
            da_head.load_state_dict(da_payload["da_head_state_dict"], strict=True)
            self.model._da_progress_head = da_head
            LOGGER.info("Loaded %d DA progress tensors from %s", len(da_head.state_dict()), da_progress_checkpoint)
        self.state_dict_count = len(state_dict)
        LOGGER.info(
            "Loaded %d tensors from checkpoint %s with scorer_mode=%s "
            "ep_exponent=%s speed_top_k=%s speed_weight=%s "
            "speed_proxy=%s curvature_weight=%s heading_change_weight=%s",
            self.state_dict_count,
            checkpoint_path,
            scorer_mode,
            ep_exponent,
            speed_top_k,
            speed_weight,
            speed_proxy,
            curvature_weight,
            heading_change_weight,
        )
        self.model.to(self.device).eval()

        if warm_up:
            zero_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            self.predict_batch(
                [
                    InferenceInput(
                        images={camera_id: zero_image for camera_id in CAMERA_IDS},
                        command_one_hot=np.array([0, 1, 0, 0], dtype=np.float32),
                        velocity_xy=np.zeros(2, dtype=np.float32),
                        acceleration_xy=np.zeros(2, dtype=np.float32),
                    )
                ]
            )
            LOGGER.info("GTRS-Dense policy warm-up completed")

    def predict_batch(self, requests: list[InferenceInput]) -> list[Prediction]:
        trajectories, debug = self.predict_batch_with_debug(requests)
        return [
            Prediction(
                trajectory=sample.copy(),
                spatial=row.get("spatial", {}),
                trace={
                    "selected_vocab_index": row["selected_index"],
                    "risk": row.get("selected_risk"),
                    "progress_m": row.get("progress_m"),
                    "fallback_reason": row.get("fallback_reason"),
                },
            )
            for sample, row in zip(trajectories, debug, strict=True)
        ]

    def predict_batch_with_debug(
        self, requests: list[InferenceInput]
    ) -> tuple[list[np.ndarray], list[dict[str, object]]]:
        """Same forward as predict_batch, plus the vocabulary scores it discards."""
        if not requests:
            return [], []

        camera_feature = torch.stack(
            [
                preprocess_images(request.images, dtype=torch.float32)
                for request in requests
            ]
        ).to(device=self.device, dtype=self.dtype)
        status_feature = torch.stack(
            [
                build_status_feature(
                    command_one_hot=request.command_one_hot,
                    velocity_xy=request.velocity_xy,
                    acceleration_xy=request.acceleration_xy,
                )
                for request in requests
            ]
        ).to(device=self.device, dtype=self.dtype)

        features = {
            "camera_feature": camera_feature,
            "status_feature": status_feature,
        }
        with torch.inference_mode():
            output = self.model(features)
            if self.spatial_execute_path and "spatial_path" in output:
                output["trajectory_route_pick"] = output["trajectory"]
                output["trajectory"] = walk_path_at_reference_arc(
                    output["trajectory_scored"], output["spatial_path"]
                )

        trajectory = output["trajectory"].detach().float().cpu().numpy()
        expected_shape = (len(requests), 40, 3)
        if trajectory.shape != expected_shape:
            raise ValueError(
                f"model trajectory must have shape {expected_shape}; "
                f"got {trajectory.shape}"
            )
        if not np.isfinite(trajectory).all():
            raise ValueError("model trajectory contains non-finite values")

        scores = output["scores"].detach().float().cpu()
        selected = output["selected_indices"].detach().cpu()
        score_shape = tuple(scores.shape)
        head_names = [
            key
            for key, value in output.items()
            if torch.is_tensor(value)
            and tuple(value.shape) == score_shape
            and key != "scores"
        ]
        heads = {
            key: output[key].detach().float().cpu()
            for key in head_names
        }
        spatial = {
            key: output[key].detach().cpu().numpy()
            for key in SPATIAL_OUTPUT_KEYS
            if key in output
        }
        debug_rows: list[dict[str, object]] = []
        trajectories: list[np.ndarray] = []
        for index, sample in enumerate(trajectory):
            row: dict[str, object] = {
                "selected_index": int(selected[index]),
                "scores": scores[index].numpy().astype(np.float32, copy=True),
                "candidate_heads": {
                    key: value[index].numpy().astype(np.float16, copy=True)
                    for key, value in heads.items()
                },
            }
            if spatial:
                row["spatial"] = {
                    key: value[index].copy() for key, value in spatial.items()
                }
                if "progress_s4" in spatial:
                    progress = float(spatial["progress_s4"][index])
                    if np.isfinite(progress):
                        row["progress_m"] = progress
                    else:
                        row["fallback_reason"] = "invalid_progress"
                if "spatial_route_pool_size" in spatial and int(spatial["spatial_route_pool_size"][index]) == 0:
                    row.setdefault("fallback_reason", "empty_route_pool")
            if "no_at_fault_collisions" in heads and "drivable_area_compliance" in heads:
                chosen = int(selected[index])
                nc = float(heads["no_at_fault_collisions"][index, chosen].sigmoid())
                dac = float(heads["drivable_area_compliance"][index, chosen].sigmoid())
                row["selected_risk"] = {"collision_probability": 1.0 - nc, "drivable_failure_probability": 1.0 - dac}
            debug_rows.append(row)
            trajectories.append(sample.copy())
        return trajectories, debug_rows
