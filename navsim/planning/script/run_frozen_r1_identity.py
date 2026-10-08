"""Identity check: frozen R1 trajectories vs the saved Navhard predictions.

Does not rescore the full split and does not write the official EPDMS csvs.
"""

import json
import os
import pickle
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataloader import default_collate

from nuplan.planning.script.builders.logging_builder import build_logger

from navsim.agents.gtrs_dense.prefix_progress import prefix_utility_log_score
from navsim.common.dataloader import SceneLoader

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score_gpu"


def _formula_self_check() -> None:
    utility = torch.tensor([[[0.2, -0.4, 0.7, 1.5], [1.0, 0.0, -1.0, 3.0]]])
    prefix = torch.tensor([[[0.5, -0.5, 1.0, 0.0], [0.25, 0.25, 0.25, 0.25]]])
    prefix_score = F.logsigmoid(prefix).mean(-1)
    got = prefix_utility_log_score(utility, prefix_score)
    expected = F.logsigmoid(utility[..., :3]).sum(-1) + prefix_score
    if not torch.equal(got, expected):
        raise RuntimeError("S_R1 formula drifted from route + NC + DAC + prefix mean")


def _poses(saved) -> np.ndarray:
    trajectory = saved["trajectory"]
    poses = trajectory.poses if hasattr(trajectory, "poses") else trajectory
    return np.asarray(poses, dtype=np.float64)


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _features(agent, scene_loader, token):
    scene = scene_loader.get_scene_from_token(token)
    features = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(scene.get_agent_input()))
    return default_collate([features])


def _original_scores(prediction):
    return (
        0.03 * prediction["imi"].softmax(-1).log()
        + 0.1 * prediction["traffic_light_compliance"].sigmoid().log()
        + 0.1 * prediction["no_at_fault_collisions"].sigmoid().log()
        + 0.9 * prediction["drivable_area_compliance"].sigmoid().log()
        + 0.2 * prediction["driving_direction_compliance"].sigmoid().log()
        + 6.0 * (
            7.0 * prediction["time_to_collision_within_bound"].sigmoid()
            + 7.0 * prediction["ego_progress"].sigmoid()
            + 3.0 * prediction["lane_keeping"].sigmoid()
        ).log()
    )


def _forward(agent, features, device):
    agent.eval()
    with torch.no_grad():
        return agent.forward(_to_device(features, device))


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    _formula_self_check()
    if not torch.cuda.is_available():
        raise RuntimeError("identity forward needs one GPU")
    device = torch.device("cuda:0")
    n_stage1 = int(os.environ.get("GTRS_IDENTITY_STAGE1", "24"))
    n_stage2 = int(os.environ.get("GTRS_IDENTITY_STAGE2", "24"))
    predictions_path = Path(os.environ["GTRS_R1_PREDICTIONS"])
    report_path = Path(os.environ.get("GTRS_IDENTITY_REPORT", str(Path(cfg.output_dir) / "identity_report.json")))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    r1_agent = instantiate(cfg.agent)
    r1_agent.initialize()
    if r1_agent.model._prefix_progress_head is None or r1_agent.model._candidate_utility_head is None:
        raise RuntimeError("R1 checkpoint was loaded without the frozen selector heads")

    vov_cfg = OmegaConf.create(OmegaConf.to_container(cfg.agent, resolve=True))
    vov_cfg.checkpoint_path = os.environ["GTRS_VOV_CHECKPOINT"]
    vov_cfg.config.spatial_path = False
    vov_cfg.config.candidate_utility = False
    vov_cfg.config.prefix_progress = False
    vov_agent = instantiate(vov_cfg)
    vov_agent.initialize()
    if vov_agent.model._candidate_utility_head is not None or vov_agent.model._prefix_progress_head is not None:
        raise RuntimeError("VoV load constructed the R1 selector")

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=r1_agent.get_sensor_config(),
    )
    stage1 = scene_loader.tokens_stage_one[:n_stage1]
    stage2 = list(scene_loader.synthetic_scenes.keys())[:n_stage2]
    if len(stage1) < n_stage1 or len(stage2) < n_stage2:
        raise RuntimeError(f"subset too small: stage1={len(stage1)} stage2={len(stage2)}")
    tokens = [(token, "stage1") for token in stage1] + [(token, "stage2") for token in stage2]

    with predictions_path.open("rb") as handle:
        saved = pickle.load(handle)
    missing = [token for token, _ in tokens if token not in saved]
    if missing:
        raise RuntimeError(f"{len(missing)} tokens missing from predictions, e.g. {missing[:3]}")

    rows = []
    max_abs = 0.0
    for token, stage in tokens:
        prediction = _forward(r1_agent, _features(r1_agent, scene_loader, token), device)
        got = prediction["trajectory"][0].detach().float().cpu().numpy()
        reference = _poses(saved[token])
        if got.shape != reference.shape:
            raise RuntimeError(f"{token} pose shape {got.shape} != {reference.shape}")
        diff = float(np.max(np.abs(got.astype(np.float64) - reference)))
        max_abs = max(max_abs, diff)
        ok = bool(np.allclose(got, reference, rtol=1e-5, atol=1e-5))
        chosen = prediction["prefix_utility_log_scores"].argmax(dim=1)
        if not torch.equal(prediction["trajectory"], r1_agent.model._trajectory_head.vocab[chosen]):
            raise RuntimeError(f"{token} trajectory is not the S_R1 argmax")
        rows.append({"token": token, "stage": stage, "max_abs": diff, "match": ok})
        print(f"[identity] {stage} {token} max_abs={diff:.3e} match={ok}", flush=True)
        del prediction
        torch.cuda.empty_cache()

    matched = sum(row["match"] for row in rows)
    vov_token = stage1[0]
    vov_prediction = _forward(vov_agent, _features(vov_agent, scene_loader, vov_token), device)
    vov_index = _original_scores(vov_prediction).argmax(1)
    vov_ok = bool(torch.equal(
        vov_prediction["trajectory"], vov_agent.model._trajectory_head.vocab[vov_index]))
    if "prefix_utility_log_scores" in vov_prediction or not vov_ok:
        raise RuntimeError("VoV batch did not stay on the original 8-head argmax")

    report = {
        "predictions": str(predictions_path),
        "r1_checkpoint": os.environ["GTRS_R1_CHECKPOINT"],
        "vov_checkpoint": os.environ["GTRS_VOV_CHECKPOINT"],
        "vocab": os.environ["GTRS_VOCAB_PATH"],
        "spatial_anchor": os.environ["GTRS_SPATIAL_ANCHOR_PATH"],
        "stage1_checked": len(stage1),
        "stage2_checked": len(stage2),
        "tokens_checked": len(rows),
        "tokens_matched": matched,
        "max_abs": max_abs,
        "tolerance": {"rtol": 1e-5, "atol": 1e-5},
        "vov_token": vov_token,
        "vov_original_8head_argmax": vov_ok,
        "full_rescore": False,
        "mismatches": [row for row in rows if not row["match"]],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "tokens_checked", "tokens_matched", "stage1_checked", "stage2_checked", "max_abs",
        "vov_original_8head_argmax")}), flush=True)
    if matched != len(rows):
        raise SystemExit(f"identity mismatch {matched}/{len(rows)}; see {report_path}")


if __name__ == "__main__":
    main()
