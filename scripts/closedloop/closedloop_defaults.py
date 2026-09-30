"""Rollouts in flight per GPU for closed-loop (alpasim) evaluation launchers.

Future closed-loop runs must use the widest stable width: the highest measured scenes/min with no
OOM, the host under its CPU quota and ~8 GB left free on each card. Do not default to 1 rollout per
GPU; one rollout uses only a few GB of a 40 GB card, and 4 GPUs x 1 scene took ~2 h on full navtest.
Change the default only with a new measurement showing a wider or narrower width is faster.

Measured on 4x A100-40GB, 32-core quota, 1194 navtest scenes, steady state after the first wave:
    4x3  32 scenes/min, <=9.9 GB/card,  ~12 cores
    4x5  40 scenes/min, <=12.4 GB/card, ~20 cores, no OOM  <- default

CLOSEDLOOP_ROLLOUTS_PER_GPU overrides the width for one run.

Usage in a launcher (import instead of hardcoding a concurrency):

    import sys
    sys.path.insert(0, "<GTRS>/scripts/closedloop")
    from closedloop_defaults import LAYOUT, ROLLOUTS_PER_GPU

    print(f"layout={LAYOUT}")
    run_closed_loop(scene_ids, concurrency_per_gpu=ROLLOUTS_PER_GPU)

    # one-off override:
    #   CLOSEDLOOP_ROLLOUTS_PER_GPU=3 python my_launcher.py
"""

from __future__ import annotations

import os

MEASURED_ROLLOUTS_PER_GPU = 5
ROLLOUTS_PER_GPU = int(os.environ.get("CLOSEDLOOP_ROLLOUTS_PER_GPU") or MEASURED_ROLLOUTS_PER_GPU)
if ROLLOUTS_PER_GPU < 1:
    raise ValueError(f"CLOSEDLOOP_ROLLOUTS_PER_GPU must be >= 1, got {ROLLOUTS_PER_GPU}")
N_GPUS = 4
LAYOUT = f"{N_GPUS}x{ROLLOUTS_PER_GPU}"
