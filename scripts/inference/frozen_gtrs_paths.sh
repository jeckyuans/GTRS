#!/bin/bash
# TEMPORARY absolute paths for this frozen-R1 submission.
# They are machine-local. Replace this file before the checkout moves.
# VoV start stays on the original 8-head argmax (agent=gtrs_dense_vov).
# R1 uses agent=gtrs_dense_vov_frozen_r1 and the S_R1 selector.

set -euo pipefail

export GTRS_VOCAB_PATH=/home/ys/alpasim_ws/GTRS/traj_final/16384.npy
export GTRS_R1_CHECKPOINT=/home/ys/alpasim_ws/GTRS/outputs/vov15h_20261001/r1_route/model.ckpt
export GTRS_VOV_CHECKPOINT=/ai/ys/alpasim_ws/gtrs-assets/gtrs_dense_vov_sim_reward_navhard.ckpt
export GTRS_SPATIAL_ANCHOR_PATH=/ai/ys/navsim_exp/spatial_path_anchors_train_2m24.npz
export GTRS_R1_PREDICTIONS=/ai/ys/navsim_exp/gtrs_dense/navhard_epdms_20261006/r1/predictions.pkl
export GTRS_IDENTITY_REPORT=/ai/ys/navsim_exp/gtrs_dense/frozen_r1_identity_20261008/identity_report.json

export OPENSCENE_DATA_ROOT=/ai/ys/navsim_dataset
export NUPLAN_MAPS_ROOT=/ai/ys/navsim_dataset/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NAVSIM_EXP_ROOT=/ai/ys/navsim_exp/gtrs_dense
export NAVSIM_TRAJPDM_ROOT=/ai/ys/navsim_dataset/traj_pdm_v2

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export NAVSIM_DEVKIT_ROOT=$ROOT
export PYTHONPATH=$ROOT${PYTHONPATH:+:$PYTHONPATH}
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  exec /ai/ys/alpasim_ws/.venv_gtrs_train/bin/python \
    "$ROOT/navsim/planning/script/run_frozen_r1_identity.py" \
    "$@"
fi
