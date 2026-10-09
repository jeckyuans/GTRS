#!/usr/bin/env python3
"""Cross-track error of the controller reference against later ego poses."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from alpasim_utils.logs import async_read_pb_log

from navsim_gtrs_dense_challenge.trajectory import yaw_from_quat


def _wrap(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def _find_asl(run_root: Path, scene_id: str) -> Path:
    matches = sorted(run_root.glob(f"gpu*/rollouts/{scene_id}/*/rollout.asl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{scene_id}: found {len(matches)} rollout.asl files")
    return matches[0]


def _pose_xyyaw(pose) -> tuple[float, float, float]:
    return (
        float(pose.vec.x),
        float(pose.vec.y),
        yaw_from_quat(pose.quat),
    )


async def _load(path: Path) -> tuple[dict[int, tuple[float, float, float]], list]:
    """Ego comes from driver_ego_trajectory, which shares the controller frame.

    actor_poses are in the map frame and do not match controller.state.pose.
    """
    ego: dict[int, tuple[float, float, float]] = {}
    requests = []
    async for entry in async_read_pb_log(str(path)):
        kind = entry.WhichOneof("log_entry")
        if kind == "driver_ego_trajectory":
            for pose in entry.driver_ego_trajectory.trajectory.poses:
                ego[int(pose.timestamp_us)] = _pose_xyyaw(pose.pose)
        elif kind == "controller_request":
            requests.append(entry.controller_request)
    return ego, requests


def _ego_trace(ego: dict[int, tuple[float, float, float]]):
    times = sorted(ego)
    xs, ys, yaws = [], [], []
    for timestamp_us in times:
        x_m, y_m, yaw = ego[timestamp_us]
        xs.append(x_m)
        ys.append(y_m)
        yaws.append(yaw)
    return (
        np.asarray(times, dtype=np.int64),
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.unwrap(np.asarray(yaws, dtype=np.float64)),
    )


def _interval_errors(request, next_ts: int, ego_t, ego_x, ego_y, ego_yaw) -> dict | None:
    plan = request.planned_trajectory_in_rig
    if len(plan.poses) < 2:
        return None
    anchor_ts = int(request.state.timestamp_us)
    anchor_x = float(request.state.pose.vec.x)
    anchor_y = float(request.state.pose.vec.y)
    anchor_yaw = yaw_from_quat(request.state.pose.quat)
    plan_t = np.asarray([int(pose.timestamp_us) for pose in plan.poses], dtype=np.int64)
    plan_x = np.asarray([pose.pose.vec.x for pose in plan.poses], dtype=np.float64)
    plan_y = np.asarray([pose.pose.vec.y for pose in plan.poses], dtype=np.float64)
    plan_yaw = np.unwrap(
        np.asarray([yaw_from_quat(pose.pose.quat) for pose in plan.poses], dtype=np.float64)
    )
    stop = min(int(next_ts), int(plan_t[-1]))
    mask = (ego_t > anchor_ts) & (ego_t <= stop)
    if not np.any(mask):
        return None
    actual_t = ego_t[mask]
    dx = ego_x[mask] - anchor_x
    dy = ego_y[mask] - anchor_y
    cosine = math.cos(anchor_yaw)
    sine = math.sin(anchor_yaw)
    actual_x = cosine * dx + sine * dy
    actual_y = -sine * dx + cosine * dy
    actual_yaw = _wrap(ego_yaw[mask] - anchor_yaw)

    plan_yaw_i = _wrap(np.interp(actual_t, plan_t, plan_yaw))
    plan_x_i = np.interp(actual_t, plan_t, plan_x)
    plan_y_i = np.interp(actual_t, plan_t, plan_y)
    error_x = actual_x - plan_x_i
    error_y = actual_y - plan_y_i
    heading = np.cos(plan_yaw_i)
    side = np.sin(plan_yaw_i)
    longitudinal = heading * error_x + side * error_y
    lateral = -side * error_x + heading * error_y
    heading_error = _wrap(actual_yaw - plan_yaw_i)
    absolute_lateral = np.abs(lateral)
    return {
        "controller_request_ts": anchor_ts,
        "next_request_ts": int(next_ts),
        "n_samples": int(absolute_lateral.size),
        "max_abs_lat_tracking_m": float(absolute_lateral.max()),
        "p95_abs_lat_tracking_m": float(np.quantile(absolute_lateral, 0.95)),
        "mean_abs_lat_tracking_m": float(absolute_lateral.mean()),
        "max_abs_heading_error_deg": float(np.degrees(np.abs(heading_error).max())),
        "p95_abs_heading_error_deg": float(
            np.degrees(np.quantile(np.abs(heading_error), 0.95))
        ),
        "max_abs_long_tracking_m": float(np.abs(longitudinal).max()),
        "lateral_series_m": absolute_lateral.astype(np.float32),
        "sample_ts": actual_t.astype(np.int64),
    }


def _scene_rows(scene_id: str, cohort: str, asl_path: Path) -> list[dict]:
    ego, requests = asyncio.run(_load(asl_path))
    if not ego or not requests:
        raise RuntimeError(f"{scene_id}: missing ego trajectory or controller requests")
    ego_t, ego_x, ego_y, ego_yaw = _ego_trace(ego)
    rows = []
    for index, request in enumerate(requests):
        next_ts = (
            int(requests[index + 1].state.timestamp_us)
            if index + 1 < len(requests)
            else int(request.state.timestamp_us) + 10_000_000
        )
        stats = _interval_errors(request, next_ts, ego_t, ego_x, ego_y, ego_yaw)
        if stats is None:
            continue
        stats.pop("lateral_series_m")
        stats.pop("sample_ts")
        stats["scene_id"] = scene_id
        stats["cohort"] = cohort
        rows.append(stats)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--scene-list", required=True)
    parser.add_argument("--include-control-scenes", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_root = Path(args.run_root)
    groups = [
        ("hard_zero", Path(args.scene_list)),
        ("full_score_control", Path(args.include_control_scenes)),
    ]
    rows: list[dict] = []
    for cohort, path in groups:
        scene_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        for number, scene_id in enumerate(scene_ids, start=1):
            scene_rows = _scene_rows(scene_id, cohort, _find_asl(run_root, scene_id))
            rows.extend(scene_rows)
            if number % 20 == 0 or number == len(scene_ids):
                print(f"[tracking] {cohort} {number}/{len(scene_ids)}", flush=True)
    frame = pl.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output)
    controls = frame.filter(pl.col("cohort") == "full_score_control")
    summary = {
        "n_intervals": frame.height,
        "control_p95_of_max_lat_m": float(controls["max_abs_lat_tracking_m"].quantile(0.95))
        if controls.height
        else None,
        "control_median_of_max_lat_m": float(controls["max_abs_lat_tracking_m"].median())
        if controls.height
        else None,
        "hard_zero_median_of_max_lat_m": float(
            frame.filter(pl.col("cohort") == "hard_zero")["max_abs_lat_tracking_m"].median()
        ),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
