"""Resample the cached 1 m expert path and cluster anchors per driving command.

Writes:
  spatial_path_navtrain.npz
    tokens (N,), command (N,), spacings (4,), points (N, 4, 6, 2), mask (N, 4, 6)
  spatial_path_anchors.npz
    command (C,), modes (K,), spacings (4,), points (C, K, 4, 6, 2)

Points are absolute xy in the current ego frame. Anchors for 0.5/1.0/1.5 m are
the same clustered 2 m geometry, resampled, so one mode index is one path.
"""

import argparse
import gzip
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SPACINGS = (0.5, 1.0, 1.5, 2.0)
N_POINTS = 6
N_COMMANDS = 4
MODES_PER_COMMAND = 8


def resample(path_xy: np.ndarray, path_mask: np.ndarray) -> tuple:
    valid = np.cumprod((path_mask > 0.5).astype(np.float64))
    count = int(valid.sum())
    xy = np.asarray(path_xy[:count, :2], dtype=np.float64)
    poly = np.concatenate([np.zeros((1, 2), dtype=np.float64), xy], axis=0)
    segment = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segment)])
    points = np.zeros((len(SPACINGS), N_POINTS, 2), dtype=np.float32)
    mask = np.zeros((len(SPACINGS), N_POINTS), dtype=np.uint8)
    if arc[-1] < SPACINGS[0] - 1e-4:
        return points, mask
    for freq, spacing in enumerate(SPACINGS):
        target = spacing * np.arange(1, N_POINTS + 1)
        ok = target <= (arc[-1] + 1e-4)
        mask[freq] = ok.astype(np.uint8)
        if not ok.any():
            continue
        index = np.searchsorted(arc, target).clip(1, len(arc) - 1)
        arc0 = arc[index - 1]
        arc1 = arc[index]
        alpha = np.clip((target - arc0) / np.maximum(arc1 - arc0, 1e-6), 0.0, 1.0)
        sampled = poly[index - 1] + alpha[:, None] * (poly[index] - poly[index - 1])
        sampled[~ok] = 0.0
        points[freq] = sampled.astype(np.float32)
    return points, mask


def _as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _chunk(target_paths):
    tokens = []
    commands = []
    points = []
    masks = []
    for target_path in target_paths:
        token_dir = os.path.dirname(target_path)
        token = os.path.basename(token_dir)
        with gzip.open(target_path, "rb") as handle:
            data = pickle.load(handle)
        feature_path = os.path.join(token_dir, "sparsedrive_feature.gz")
        with gzip.open(feature_path, "rb") as handle:
            feature = pickle.load(handle)
        status = _as_numpy(feature["status_feature"]).reshape(-1)
        command = int(np.argmax(status[:N_COMMANDS])) if status[:N_COMMANDS].sum() > 0 else 0
        sampled, point_mask = resample(_as_numpy(data["path"]), _as_numpy(data["path_mask"]))
        tokens.append(token)
        commands.append(command)
        points.append(sampled)
        masks.append(point_mask)
    return tokens, commands, points, masks


def _list_targets(root: str):
    files = []
    for log_name in sorted(os.listdir(root)):
        log_path = os.path.join(root, log_name)
        if not os.path.isdir(log_path):
            continue
        for token in os.listdir(log_path):
            token_path = os.path.join(log_path, token)
            if os.path.isdir(token_path):
                files.append(os.path.join(token_path, "sparsedrive_target.gz"))
    return files


def _resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    """points (6, 2) at 2 m. Return 6 points at `spacing` along the same polyline."""
    poly = np.concatenate([np.zeros((1, 2), dtype=np.float64), points.astype(np.float64)], axis=0)
    segment = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segment)])
    target = spacing * np.arange(1, N_POINTS + 1)
    index = np.searchsorted(arc, target).clip(1, len(arc) - 1)
    arc0 = arc[index - 1]
    arc1 = arc[index]
    alpha = np.clip((target - arc0) / np.maximum(arc1 - arc0, 1e-6), 0.0, 1.0)
    sampled = poly[index - 1] + alpha[:, None] * (poly[index] - poly[index - 1])
    return sampled.astype(np.float32)


def cluster_anchors(points: np.ndarray, mask: np.ndarray, command: np.ndarray):
    from sklearn.cluster import KMeans

    # Frequency index 3 is 2.0 m, the geometry used to define a mode.
    full = mask[:, 3].sum(axis=1) == N_POINTS
    anchors = np.zeros((N_COMMANDS, MODES_PER_COMMAND, len(SPACINGS), N_POINTS, 2), dtype=np.float32)
    counts = []
    for cmd in range(N_COMMANDS):
        chosen = points[full & (command == cmd), 3]
        counts.append(int(chosen.shape[0]))
        if chosen.shape[0] == 0:
            continue
        n_clusters = min(MODES_PER_COMMAND, chosen.shape[0])
        centers = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(chosen.reshape(len(chosen), -1)).cluster_centers_
        centers = centers.reshape(n_clusters, N_POINTS, 2)
        if n_clusters < MODES_PER_COMMAND:
            extra = np.resize(centers, (MODES_PER_COMMAND, N_POINTS, 2))
            centers = extra
        for mode in range(MODES_PER_COMMAND):
            anchors[cmd, mode, len(SPACINGS) - 1] = centers[mode]
            for freq, spacing in enumerate(SPACINGS[:-1]):
                anchors[cmd, mode, freq] = _resample_polyline(centers[mode], spacing)
    for cmd, count in enumerate(counts):
        if count == 0:
            anchors[cmd] = anchors[1]
    return anchors, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/ai/ys/navsim_exp/data_cache_navtrain")
    parser.add_argument("--out", default="/ai/ys/navsim_exp/spatial_path_navtrain.npz")
    parser.add_argument("--anchors", default="/ai/ys/navsim_exp/spatial_path_anchors.npz")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    print("listing", args.root, flush=True)
    files = _list_targets(args.root)
    if not files:
        raise SystemExit(f"no sparsedrive_target.gz under {args.root}")
    print("targets", len(files), flush=True)
    chunks = [files[i:i + 256] for i in range(0, len(files), 256)]
    points = np.zeros((len(files), len(SPACINGS), N_POINTS, 2), dtype=np.float32)
    mask = np.zeros((len(files), len(SPACINGS), N_POINTS), dtype=np.uint8)
    command = np.zeros(len(files), dtype=np.uint8)
    tokens = np.empty(len(files), dtype="U32")
    row = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (chunk_tokens, chunk_cmd, chunk_pts, chunk_msk) in enumerate(pool.map(_chunk, chunks), start=1):
            n = len(chunk_tokens)
            tokens[row:row + n] = chunk_tokens
            command[row:row + n] = chunk_cmd
            points[row:row + n] = np.stack(chunk_pts)
            mask[row:row + n] = np.stack(chunk_msk)
            row += n
            if done % 20 == 0 or done == len(chunks):
                print(f"{row}/{len(files)}", flush=True)
    order = np.argsort(tokens)
    tokens = tokens[order]
    command = command[order]
    points = points[order]
    mask = mask[order]
    np.savez(
        args.out,
        tokens=tokens,
        command=command,
        spacings=np.asarray(SPACINGS, dtype=np.float32),
        points=points,
        mask=mask,
    )
    anchors, counts = cluster_anchors(points, mask, command)
    np.savez(
        args.anchors,
        command=np.arange(N_COMMANDS, dtype=np.uint8),
        modes=np.arange(MODES_PER_COMMAND, dtype=np.uint8),
        spacings=np.asarray(SPACINGS, dtype=np.float32),
        points=anchors,
        counts=np.asarray(counts, dtype=np.int64),
    )
    covered = (mask.sum(axis=-1) == N_POINTS).mean(axis=0)
    print(f"wrote {args.out} n={len(files)} full6={covered.tolist()}", flush=True)
    print(f"wrote {args.anchors} per_command={counts}", flush=True)


if __name__ == "__main__":
    main()
