"""Merge a trained C3 switch gate into the fixed C0 one-file checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import torch

from navsim_gtrs_dense_challenge.simscale_gtrs_dense.switch_gate import (
    FEATURE_NAMES,
    GATE_THRESHOLD,
)


PREFIX = "agent.model._switch_gate."
C0_SHA256 = "db77938349c71a2d316cd2801098d2fbf269205d32dfd0f666827e38b3913d9b"
EXPECTED_SOURCE = {
    "shards": "/ai/ys/navsim_exp/route_pick_val_gate_20260930",
    "nav_s4": "/ai/ys/navsim_exp/spatial_progress_s4_2m24.npz",
    "off_s4": "/ai/ys/navsim_exp/simscale_offroute/spatial_progress_s4_offroute.npz",
    "train_splits": ["tr_nav", "tr_off"],
    "validation_splits": ["val_nav", "val_off"],
}
REQUIRED_ARTIFACT_KEYS = {
    "gate_state_dict", "feature_mean", "feature_std", "feature_names",
    "threshold", "label_convention", "source",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trained_gate(path: Path) -> dict:
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Only the explicitly supplied, trusted training artifact uses this fallback.
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or set(artifact) != REQUIRED_ARTIFACT_KEYS:
        raise ValueError("gate artifact has unexpected or missing fields")
    if tuple(artifact["feature_names"]) != FEATURE_NAMES:
        raise ValueError("gate feature order differs from deployment")
    if artifact["threshold"] != GATE_THRESHOLD or artifact["label_convention"] != "1_choose_c0":
        raise ValueError("gate decision convention differs from deployment")
    if artifact["source"] != EXPECTED_SOURCE:
        raise ValueError("gate artifact training source differs from frozen C3 contract")
    mean = torch.as_tensor(artifact["feature_mean"], dtype=torch.float32)
    std = torch.as_tensor(artifact["feature_std"], dtype=torch.float32)
    if mean.shape != (13,) or std.shape != (13,):
        raise ValueError("gate normalization must have 13 entries")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("gate normalization is non-finite or has non-positive std")
    state = artifact["gate_state_dict"]
    expected = {"0.weight": (32, 13), "0.bias": (32,), "2.weight": (1, 32), "2.bias": (1,)}
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("gate model has unexpected or missing tensors")
    for key, shape in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"gate tensor {key} is invalid")
    artifact["feature_mean"] = mean
    artifact["feature_std"] = std
    return artifact


def package(base_model: Path, gate_path: Path, output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    base_sha = sha256(base_model)
    if base_sha != C0_SHA256:
        raise ValueError(f"C3 requires canonical C0 model SHA256 {C0_SHA256}; got {base_sha}")
    base = torch.load(base_model, map_location="cpu", weights_only=True)
    if not isinstance(base, dict) or set(base) != {"state_dict"}:
        raise ValueError("C0 base must contain only state_dict")
    state = base["state_dict"]
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError("C0 base state_dict is invalid")
    if not any(key.startswith("agent.model._spatial_head.") for key in state):
        raise ValueError("C0 base lacks the fixed spatial head")
    if any(key.startswith(PREFIX) or key.startswith("agent.model._da_progress_head.") for key in state):
        raise ValueError("base already has a switch gate or DA progress head")
    artifact = load_trained_gate(gate_path)
    merged = dict(state)
    for key, value in artifact["gate_state_dict"].items():
        merged[PREFIX + "mlp." + key] = value
    merged[PREFIX + "feature_mean"] = artifact["feature_mean"]
    merged[PREFIX + "feature_std"] = artifact["feature_std"]
    metadata = {
        "feature_names": list(FEATURE_NAMES),
        "threshold": str(GATE_THRESHOLD),
        "label_convention": "1_choose_c0",
        "source": artifact["source"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model.ckpt"
    temporary = output_dir / "model.ckpt.partial"
    torch.save({"state_dict": merged, "switch_gate_metadata": metadata}, temporary)
    temporary.replace(checkpoint)
    source_config = json.loads((base_model.parent / "inference.json").read_text())
    source_config["GTRS_CHECKPOINT_PATH"] = str(checkpoint)
    (output_dir / "inference.json").write_text(
        json.dumps(source_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source_root = Path(__file__).resolve().parents[1]
    manifest = {
        "c0_model": {"path": str(base_model), "sha256": base_sha},
        "gate_artifact": {"path": str(gate_path), "sha256": sha256(gate_path)},
        "model_checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "source_sha256": {
            str(path.relative_to(source_root)): sha256(path)
            for path in sorted(source_root.rglob("*.py"))
        },
    }
    (output_dir / "source_sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"packaged C0 with {len(artifact['gate_state_dict'])} gate parameters into {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    package(args.base_model, args.gate, args.output_dir)


if __name__ == "__main__":
    main()
