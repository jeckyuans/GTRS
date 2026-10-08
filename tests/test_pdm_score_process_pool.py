"""CPU-only checks for sliced EPDMS tasks and spawn-pool selection.

python -m pytest tests/test_pdm_score_process_pool.py
"""

import os

from omegaconf import OmegaConf

from navsim.planning.script.run_pdm_score_gpu_v2 import (
    build_pdm_score_data_points,
    chunk_list,
    collect_pdm_score_rows,
    merge_model_trajectories,
    resolve_pdm_worker_count,
    run_pdm_score,
    score_with_spawn_pool,
    slice_model_trajectories,
)


def test_slice_drops_unused_keys_and_other_tokens():
    trajectory = {"poses": [1.0, 2.0]}
    predictions = {
        "other": {"trajectory": "skip-me", "imi": [9]},
        "keep": {"trajectory": trajectory, "imi": [1, 2], "scores": [0] * 7},
        "also": {"trajectory": "later", "imi": [3]},
    }

    sliced = slice_model_trajectories(predictions, ["missing", "keep", "also"])

    assert list(sliced) == ["keep", "also"]
    assert sliced["keep"] == {"trajectory": trajectory}
    assert sliced["keep"]["trajectory"] is trajectory
    assert "imi" not in sliced["keep"] and "scores" not in sliced["keep"]
    assert predictions["keep"]["imi"] == [1, 2]
    assert slice_model_trajectories({"bad": object(), "plain": {"nope": 1}}, ["bad", "plain"]) == {}


def test_each_log_keeps_only_its_trajectories():
    predictions = {
        "a1": {"trajectory": "a1", "imi": 1},
        "a2": {"trajectory": "a2", "scores": 2},
        "b1": {"trajectory": "b1", "imi": 3},
    }
    cfg = OmegaConf.create({"worker": {"use_process_pool": False}})

    points = build_pdm_score_data_points(cfg, predictions, {"logA": ["a1", "a2"], "logB": ["b1"]})

    assert [point["log_file"] for point in points] == ["logA", "logB"]
    assert points[0]["tokens"] == ["a1", "a2"]
    assert points[0]["model_trajectory"] == {"a1": {"trajectory": "a1"}, "a2": {"trajectory": "a2"}}
    assert points[1]["model_trajectory"] == {"b1": {"trajectory": "b1"}}
    assert "imi" not in points[0]["model_trajectory"]["a1"]
    assert predictions["b1"]["imi"] == 3


def test_merge_unions_per_log_slices():
    args = [
        {"model_trajectory": {"a": {"trajectory": "ta"}}},
        {"model_trajectory": {"b": {"trajectory": "tb"}}},
    ]

    assert merge_model_trajectories(args) == {"a": {"trajectory": "ta"}, "b": {"trajectory": "tb"}}


def test_worker_count_honors_override_and_null():
    from nuplan.planning.utils.multithreading.worker_pool import WorkerResources

    assert resolve_pdm_worker_count(OmegaConf.create({"worker": {"max_workers": 32}})) == 32
    cfg = OmegaConf.create({"worker": {"max_workers": None}})
    assert resolve_pdm_worker_count(cfg) == int(WorkerResources.current_node_cpu_count())


def test_thread_path_keeps_worker_map_and_sliced_tasks():
    predictions = {"a": {"trajectory": "ta", "imi": [1, 2, 3]}}
    cfg = OmegaConf.create({"worker": {"use_process_pool": False, "max_workers": 16}})
    data_points = build_pdm_score_data_points(cfg, predictions, {"log": ["a"]})
    seen = {}

    def worker_map_fn(worker, fn, items):
        seen["worker"] = worker
        seen["fn"] = fn
        seen["items"] = items
        return ["thread-row"]

    def spawn_fn(cfg, items):
        raise AssertionError("spawn pool should stay unused when use_process_pool is false")

    rows = collect_pdm_score_rows(
        cfg,
        data_points,
        worker_map_fn=worker_map_fn,
        build_worker_fn=lambda _cfg: "thread-worker",
        spawn_fn=spawn_fn,
    )

    assert rows == ["thread-row"]
    assert seen["worker"] == "thread-worker"
    assert seen["fn"] is run_pdm_score
    assert seen["items"][0]["model_trajectory"] == {"a": {"trajectory": "ta"}}


def test_missing_process_pool_flag_stays_on_worker_map():
    cfg = OmegaConf.create({"worker": {"threads_per_node": None}})
    seen = {}

    def worker_map_fn(worker, fn, items):
        seen["called"] = True
        return ["ray-row"]

    rows = collect_pdm_score_rows(
        cfg,
        [{"log_file": "L"}],
        worker_map_fn=worker_map_fn,
        build_worker_fn=lambda _cfg: "ray-worker",
        spawn_fn=lambda _cfg, _items: (_ for _ in ()).throw(AssertionError("spawn")),
    )

    assert rows == ["ray-row"]
    assert seen["called"] is True


def test_process_pool_flag_selects_spawn_and_skips_worker_map():
    cfg = OmegaConf.create({"worker": {"use_process_pool": True, "max_workers": 32}})

    def unused(*_args, **_kwargs):
        raise AssertionError("thread/ray worker_map should not run")

    rows = collect_pdm_score_rows(
        cfg,
        [{"log_file": "L"}],
        worker_map_fn=unused,
        build_worker_fn=unused,
        spawn_fn=lambda _cfg, items: ["spawned", items[0]["log_file"]],
    )

    assert rows == ["spawned", "L"]


def test_spawn_pool_uses_spawn_context_and_preserves_chunk_order():
    cfg = OmegaConf.create({"worker": {"use_process_pool": True, "max_workers": 2}})
    data_points = [{"log_file": f"log{i}", "tokens": [str(i)], "model_trajectory": {}} for i in range(5)]
    created = {}

    class _FakeExecutor:
        def __init__(self, max_workers, mp_context):
            created["executor"] = self
            self.max_workers = max_workers
            self.mp_context = mp_context

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, chunks):
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            self.fn = fn
            self.chunks = list(chunks)
            return [[chunk[0]["log_file"]] for chunk in self.chunks]

    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "gpu-test-0"
    try:
        rows = score_with_spawn_pool(cfg, data_points, executor_factory=_FakeExecutor)
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "gpu-test-0"
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous

    executor = created["executor"]
    expected_chunks = chunk_list(data_points, 2)
    assert executor.mp_context.get_start_method() == "spawn"
    assert executor.max_workers == len(expected_chunks)
    assert executor.fn.__module__ == "navsim.planning.script.run_pdm_score_gpu_v2"
    assert executor.fn.__name__ == "run_pdm_score"
    assert executor.chunks == expected_chunks
    assert rows == [chunk[0]["log_file"] for chunk in expected_chunks]


def test_spawn_pool_restores_cuda_env_when_mapping_fails():
    cfg = OmegaConf.create({"worker": {"use_process_pool": True, "max_workers": 2}})

    class _FailingExecutor:
        def __init__(self, max_workers, mp_context):
            self.max_workers = max_workers
            self.mp_context = mp_context

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, chunks):
            raise RuntimeError("worker failed")

    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "gpu-test-1"
    try:
        try:
            score_with_spawn_pool(
                cfg,
                [{"log_file": "only"}],
                executor_factory=_FailingExecutor,
            )
            raised = False
        except RuntimeError as exc:
            raised = str(exc) == "worker failed"
        assert raised
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "gpu-test-1"
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def test_nonpositive_workers_run_in_process_without_an_executor():
    cfg = OmegaConf.create({"worker": {"use_process_pool": True, "max_workers": 0}})
    seen = {}

    def score_fn(args):
        seen["args"] = args
        return ["local"]

    def executor_factory(*_args, **_kwargs):
        raise AssertionError("executor should not be constructed")

    rows = score_with_spawn_pool(
        cfg,
        [{"log_file": "only"}],
        executor_factory=executor_factory,
        score_fn=score_fn,
    )

    assert rows == ["local"]
    assert seen["args"] == [{"log_file": "only"}]


def test_empty_inputs_skip_the_pool():
    cfg = OmegaConf.create({"worker": {"use_process_pool": True, "max_workers": 4}})

    def executor_factory(*_args, **_kwargs):
        raise AssertionError("executor should not be constructed")

    assert score_with_spawn_pool(cfg, [], executor_factory=executor_factory, score_fn=lambda _args: ["no"]) == []
