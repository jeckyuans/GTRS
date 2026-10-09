#!/usr/bin/env python3
"""Replay saved ASL observations through the frozen GTRS driver.

The returned trajectory is built with the production plan cache and
trajectory sampler. Vocabulary scores are written beside that trajectory
and are not fed back into it.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from alpasim_utils.logs import async_read_pb_log

from navsim_gtrs_dense_challenge.driver import SessionState, decode_driver_image
from navsim_gtrs_dense_challenge.navigation import command_from_route
from navsim_gtrs_dense_challenge.policy import GTRSDensePolicy, InferenceInput
from navsim_gtrs_dense_challenge.trajectory import (
    build_trajectory_from_plan,
    cached_plan_covers_query,
    make_cached_plan,
    yaw_from_quat,
)

FROZEN_COMMIT = "1839a9457a0304956e712090a16a5593d5c9ab54"
FROZEN_EVAL_SHA = "b94c27f3258957ebab7b9c678fd3e974412971d3ced448e8f7df41ef87b029d7"
FROZEN_SUMMARY_SHA = "b46c7e5c988dc24781cfbead9e656eacb9bd00e1446f2a43a87cde6f1dc9eb71"
FROZEN_METRICS_SHA = "774e1004c4982c7864b1fb803d3c51a95d5dc0b099c1925ca6818e33464dc00e"
FROZEN_CKPT_SHA = "8dad0395332ccd844785cbfc7c9e24cb3f8d8dbf5cb9ca7f8f8dc75478fcf409"
DEFAULT_VOCAB = (
    "/home/ys/alpasim_ws/alpasim/e2e_challenge/"
    "sample_submission_simscale_navsim_gtrs_dense/assets/gtrs_dense/navsim_16384.npy"
)
SIM_PYTHON = "/ai/ys/alpasim_ws/.venv_sim/bin/python"
XY_GATE_M = 0.01
YAW_GATE_DEG = 0.5
MATCH_FRACTION = 0.99


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preflight(args: argparse.Namespace) -> None:
    repo = Path("/home/ys/alpasim_ws/alpasim")
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != FROZEN_COMMIT:
        raise SystemExit(f"AlpaSim HEAD {head} != {FROZEN_COMMIT}")
    run = Path(args.run_root)
    for gpu in range(4):
        digest = _sha256(run / f"gpu{gpu}" / "eval-config.yaml")
        if digest != FROZEN_EVAL_SHA:
            raise SystemExit(f"gpu{gpu} eval-config sha256 {digest}")
    if _sha256(run / "aggregate" / "results-summary.json") != FROZEN_SUMMARY_SHA:
        raise SystemExit("results-summary.json sha256 mismatch")
    if _sha256(run / "aggregate" / "metrics_results.txt") != FROZEN_METRICS_SHA:
        raise SystemExit("metrics_results.txt sha256 mismatch")
    if _sha256(Path(args.checkpoint)) != FROZEN_CKPT_SHA:
        raise SystemExit("checkpoint sha256 mismatch")


def _load_failure_index(path: Path) -> dict[str, dict]:
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
    elif path.suffix == ".parquet":
        payload = subprocess.check_output(
            [
                SIM_PYTHON,
                "-c",
                "import json,sys,polars as pl; "
                "print(pl.read_parquet(sys.argv[1]).write_json())",
                str(path),
            ],
            text=True,
        )
        rows = json.loads(payload)
    else:
        raise SystemExit(f"unsupported failure index {path}")
    index = {}
    for row in rows:
        index[row["scene_id"]] = row
    return index


def _find_asl(run_root: Path, scene_id: str) -> Path:
    matches = sorted(run_root.glob(f"gpu*/rollouts/{scene_id}/*/rollout.asl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{scene_id}: found {len(matches)} rollout.asl files")
    return matches[0]


def _wrap_rad(delta: float) -> float:
    return math.atan2(math.sin(delta), math.cos(delta))


def _trajectory_error(produced, recorded) -> tuple[float, float, bool]:
    if len(produced.poses) != len(recorded.poses) or not produced.poses:
        return 1.0e3, 180.0, False
    xy_max = 0.0
    yaw_max = 0.0
    for left, right in zip(produced.poses, recorded.poses, strict=True):
        if int(left.timestamp_us) != int(right.timestamp_us):
            return 1.0e3, 180.0, False
        dx = left.pose.vec.x - right.pose.vec.x
        dy = left.pose.vec.y - right.pose.vec.y
        xy_max = max(xy_max, math.hypot(dx, dy))
        yaw_max = max(
            yaw_max,
            abs(_wrap_rad(yaw_from_quat(left.pose.quat) - yaw_from_quat(right.pose.quat))),
        )
    return xy_max, math.degrees(yaw_max), True


def _drive_once(session: SessionState, request, policy: GTRSDensePolicy, time_scale: float):
    """Mirror NavsimGTRSDenseDriver.drive without the gRPC server."""
    time_now_us = int(request.time_now_us)
    time_query_us = int(request.time_query_us)
    current_pose, current_velocity, _current_acceleration = session.ego_snapshot(
        time_now_us
    )
    image_timestamp_us = None
    inference_input = None
    inference_pose = None
    try:
        image_timestamp_us, images = session.synchronized_images(time_now_us)
        if image_timestamp_us == current_pose.timestamp_us:
            inference_pose = current_pose
            velocity = current_velocity
            acceleration = _current_acceleration
        else:
            inference_pose, velocity, acceleration = session.ego_snapshot(
                image_timestamp_us
            )
        inference_input = InferenceInput(
            images=dict(images),
            command_one_hot=session.command_one_hot.copy(),
            velocity_xy=velocity.copy(),
            acceleration_xy=acceleration.copy(),
        )
    except LookupError:
        inference_input = None

    timestamp_is_new = image_timestamp_us is not None and (
        session.last_inference_timestamp_us is None
        or image_timestamp_us > session.last_inference_timestamp_us
    )
    should_infer = (
        inference_input is not None
        and inference_pose is not None
        and timestamp_is_new
        and session.inference_inflight_timestamp_us is None
    )
    debug = None
    fresh_plan = None
    if should_infer:
        session.last_inference_timestamp_us = image_timestamp_us
        trajectories, debug_rows = policy.predict_batch_with_debug([inference_input])
        debug = debug_rows[0]
        fresh_plan = make_cached_plan(
            image_timestamp_us,
            inference_pose,
            trajectories[0],
            trajectory_time_scale=time_scale,
        )
        session.cached_plan = fresh_plan

    response_plan = session.cached_plan
    used_cache = fresh_plan is None and cached_plan_covers_query(
        response_plan, time_now_us, time_query_us
    )
    trajectory = build_trajectory_from_plan(
        response_plan,
        current_pose,
        time_now_us,
        time_query_us,
        fallback_speed_mps=float(np.linalg.norm(current_velocity)),
    )
    return {
        "time_now_us": time_now_us,
        "time_query_us": time_query_us,
        "inference_timestamp_us": image_timestamp_us,
        "fresh_inference": fresh_plan is not None,
        "used_cache": used_cache,
        "trajectory": trajectory,
        "debug": debug,
    }


def _save_debug(directory: Path, scene_id: str, timestamp_us: int, debug: dict) -> str:
    scene_dir = directory / "npz" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    path = scene_dir / f"{timestamp_us}.npz"
    arrays = {
        "version": np.asarray([1], dtype=np.int16),
        "selected_index": np.asarray([debug["selected_index"]], dtype=np.int32),
        "scores": np.asarray(debug["scores"], dtype=np.float32),
    }
    for name, values in debug["candidate_heads"].items():
        arrays[f"head__{name}"] = np.asarray(values, dtype=np.float16)
    np.savez_compressed(path, **arrays)
    return str(path)


async def _replay_scene(
    asl_path: Path,
    policy: GTRSDensePolicy,
    failure_us: int,
    window_s: float,
    output: Path,
    scene_id: str,
) -> list[dict]:
    session = SessionState()
    pending = None
    rows: list[dict] = []
    saved_inference: set[int] = set()
    window_us = int(round(window_s * 1_000_000))
    async for entry in async_read_pb_log(str(asl_path)):
        kind = entry.WhichOneof("log_entry")
        if kind == "driver_camera_image":
            image = entry.driver_camera_image.camera_image
            decoded = decode_driver_image(image.logical_id, image.image_bytes)
            if decoded is not None:
                session.add_image(image.logical_id, int(image.frame_end_us), decoded)
        elif kind == "driver_ego_trajectory":
            message = entry.driver_ego_trajectory
            session.add_egomotion(
                list(message.trajectory.poses),
                list(message.dynamic_states),
            )
        elif kind == "route_request":
            session.command_one_hot = command_from_route(entry.route_request.route)
        elif kind == "driver_request":
            if pending is not None:
                raise RuntimeError(f"{scene_id}: driver_request without a return")
            pending = entry.driver_request
        elif kind == "driver_return":
            if pending is None:
                raise RuntimeError(f"{scene_id}: driver_return without a request")
            result = _drive_once(session, pending, policy, 1.0)
            xy_m, yaw_deg, aligned = _trajectory_error(
                result["trajectory"], entry.driver_return.trajectory
            )
            in_window = (
                failure_us - window_us
                <= result["time_now_us"]
                <= failure_us
            )
            sidecar = None
            debug = result["debug"]
            inference_us = result["inference_timestamp_us"]
            if in_window and debug is not None and inference_us not in saved_inference:
                sidecar = _save_debug(output, scene_id, int(inference_us), debug)
                saved_inference.add(int(inference_us))
            if in_window or result["fresh_inference"]:
                scores = None if debug is None else debug["scores"]
                selected = None if debug is None else int(debug["selected_index"])
                top32 = []
                selected_score = None
                if scores is not None and selected is not None:
                    order = np.argsort(-scores)[:32]
                    top32 = [int(index) for index in order]
                    selected_score = float(scores[selected])
                rows.append(
                    {
                        "scene_id": scene_id,
                        "query_timestamp_us": result["time_now_us"],
                        "inference_timestamp_us": inference_us,
                        "fresh_inference": result["fresh_inference"],
                        "used_cache": result["used_cache"],
                        "in_failure_window": in_window,
                        "selected_index": selected,
                        "selected_score": selected_score,
                        "top32_indices": top32,
                        "plan_xy_max_error_m": xy_m,
                        "plan_yaw_max_error_deg": yaw_deg,
                        "plan_timestamps_aligned": aligned,
                        "score_sidecar": sidecar,
                    }
                )
            pending = None
    if pending is not None:
        raise RuntimeError(f"{scene_id}: trailing driver_request")
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _gate(rows: list[dict]) -> dict:
    compared = [row for row in rows if row["fresh_inference"] or row["in_failure_window"]]
    if not compared:
        raise SystemExit("no replayed frames to validate")
    bad = [
        row
        for row in compared
        if row["plan_xy_max_error_m"] > XY_GATE_M
        or row["plan_yaw_max_error_deg"] > YAW_GATE_DEG
    ]
    report = {
        "compared_frames": len(compared),
        "mismatched_frames": len(bad),
        "match_fraction": 1.0 - len(bad) / len(compared),
        "max_xy_error_m": max(row["plan_xy_max_error_m"] for row in compared),
        "max_yaw_error_deg": max(row["plan_yaw_max_error_deg"] for row in compared),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--failure-index", required=True)
    parser.add_argument("--scene-list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-before-s", type=float, default=2.0)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _preflight(args)

    failures = _load_failure_index(Path(args.failure_index))
    scene_ids = [
        line.strip()
        for line in Path(args.scene_list).read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        scene_ids = scene_ids[: args.limit]
    missing = [scene_id for scene_id in scene_ids if scene_id not in failures]
    if missing:
        raise SystemExit(f"{len(missing)} scenes missing from the failure index")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    done_path = output / "scenes_done.txt"
    done = set(done_path.read_text().split()) if done_path.is_file() else set()
    frames_path = output / "frames.jsonl"

    policy = GTRSDensePolicy(
        args.checkpoint,
        args.vocab,
        device=args.device,
        scorer_mode="nc_dac_ep",
        ep_exponent=3.0,
        speed_top_k=64,
        speed_weight=3.0,
        speed_proxy="longitudinal",
        curvature_weight=0.0,
        heading_change_weight=0.0,
        warm_up=False,
        backbone_type="resnet",
    )
    policy.model.eval()

    all_rows: list[dict] = []
    if frames_path.is_file() and done:
        all_rows.extend(json.loads(line) for line in frames_path.read_text().splitlines() if line)

    run_root = Path(args.run_root)
    for scene_id in scene_ids:
        if scene_id in done:
            continue
        failure_us = int(round(float(failures[scene_id]["failure_frame_us"])))
        asl_path = _find_asl(run_root, scene_id)
        rows = asyncio.run(
            _replay_scene(
                asl_path,
                policy,
                failure_us,
                args.window_before_s,
                output,
                scene_id,
            )
        )
        _write_jsonl(frames_path, rows)
        all_rows.extend(rows)
        with done_path.open("a") as handle:
            handle.write(scene_id + "\n")
        done.add(scene_id)
        fresh = sum(row["fresh_inference"] for row in rows)
        window = sum(row["in_failure_window"] for row in rows)
        print(
            f"[rescore] {len(done)}/{len(scene_ids)} {scene_id} "
            f"fresh={fresh} window={window}",
            flush=True,
        )

    report = _gate(all_rows)
    (output / "replay_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if report["match_fraction"] < MATCH_FRACTION:
        raise SystemExit(
            f"replay match fraction {report['match_fraction']:.4f} "
            f"is below {MATCH_FRACTION}"
        )


if __name__ == "__main__":
    os.environ.setdefault("TORCH_NUM_THREADS", "1")
    main()
