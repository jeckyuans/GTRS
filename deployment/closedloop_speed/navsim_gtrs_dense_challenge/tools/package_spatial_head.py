"""Package the released VoV scorer and a spatial head into one deployment checkpoint.

Only ``agent.model._spatial_head.*`` is copied from the head checkpoint. The
released scorer's original state_dict is otherwise unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import torch


PREFIX = "agent.model._spatial_head."
CONFIG = {
    "GTRS_BACKBONE": "vov",
    "GTRS_SCORER_MODE": "nc_dac_ep",
    "GTRS_EP_EXPONENT": "3.0",
    "GTRS_SPEED_ENHANCEMENT": "1",
    "GTRS_SPEED_TOP_K": "64",
    "GTRS_SPEED_WEIGHT": "3.0",
    "GTRS_SPATIAL_PATH": "1",
    "GTRS_SPATIAL_PLAN_TOPK": "32",
    "GTRS_SPATIAL_RERANK_SCORE_MARGIN": "1.0",
    "GTRS_SPATIAL_ROUTE_ARC_GATE": "1",
    "GTRS_SPATIAL_RERANK_MIN_REF_DEV": "0.0",
    "GTRS_SPATIAL_RERANK_MIN_IMPROVEMENT": "0.0",
    "GTRS_SPATIAL_REFERENCE_SAFETY_GATE": "1",
    "GTRS_SPATIAL_EXECUTE_PATH": "0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict(path: Path, *, trusted_legacy: bool = False) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        if not trusted_legacy:
            raise
        # The fixed step544 Lightning checkpoint predates weights_only support
        # for some of its metadata. This fallback is only for the caller's
        # explicitly supplied, trusted training artifact.
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"{path} lacks state_dict")
    state = payload["state_dict"]
    if not isinstance(state, dict) or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError(f"{path} has an invalid state_dict")
    return state


def package(released: Path, path_head: Path, output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    released_state = state_dict(released)
    head_state = state_dict(path_head, trusted_legacy=True)
    if any(key.startswith(PREFIX) for key in released_state):
        raise ValueError("released checkpoint already contains a spatial head")
    spatial = {key: value for key, value in head_state.items() if key.startswith(PREFIX)}
    if not spatial or PREFIX + "anchors" not in spatial:
        raise ValueError("spatial checkpoint lacks path head or anchors")
    merged = dict(released_state)
    merged.update(spatial)
    if len(merged) != len(released_state) + len(spatial):
        raise AssertionError("a spatial key overwrote a released scorer key")

    checkpoint = output_dir / "model.ckpt"
    temporary = output_dir / "model.ckpt.partial"
    torch.save({"state_dict": merged}, temporary)
    temporary.replace(checkpoint)
    inference = dict(CONFIG)
    inference["GTRS_CHECKPOINT_PATH"] = str(checkpoint)
    (output_dir / "inference.json").write_text(
        json.dumps(inference, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    source_root = Path(__file__).resolve().parents[1]
    source_files = sorted(source_root.rglob("*.py"))
    manifest = {
        "released_checkpoint": {"path": str(released), "sha256": sha256(released)},
        "spatial_head_checkpoint": {"path": str(path_head), "sha256": sha256(path_head)},
        "model_checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "spatial_tensors": len(spatial),
        "released_tensors": len(released_state),
        "source_sha256": {
            str(path.relative_to(source_root)): sha256(path) for path in source_files
        },
    }
    (output_dir / "source_sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"packaged {len(spatial)} spatial tensors into {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--released", type=Path, required=True)
    parser.add_argument("--path-head", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    package(args.released, args.path_head, args.output_dir)


if __name__ == "__main__":
    main()
