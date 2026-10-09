"""Verify overlap, response ownership and bounded buffers without any CUDA."""
import importlib.util
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKERS = [ROOT / "deployment/closedloop_speed/navsim_gtrs_dense_challenge/batch_worker.py"]


@pytest.fixture(params=WORKERS, ids=["gtrs"])
def worker_class(request):
    package = "_pipeline_test_" + request.param.parent.name
    parent = types.ModuleType(package)
    parent.__path__ = [str(request.param.parent)]
    sys.modules[package] = parent
    policy = types.ModuleType(package + ".policy")
    policy.InferenceInput = object
    policy.Prediction = object
    sys.modules[package + ".policy"] = policy
    spec = importlib.util.spec_from_file_location(package + ".batch_worker", request.param)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.BatchWorker


class Policy:
    def predict_batch(self, requests):
        raise AssertionError("raw inference must not run on the staged path")

    def prepare_request(self, request):
        return request * 10

    def predict_prepared_batch(self, requests):
        return requests


def test_independent_requests_prepare_concurrently(worker_class):
    rendezvous = threading.Barrier(2)
    class ConcurrentPolicy(Policy):
        def prepare_request(self, request):
            rendezvous.wait(timeout=2)
            return request * 10
    worker = worker_class(ConcurrentPolicy(), max_batch_size=2, batch_window_s=0.1)
    worker.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker.predict, request, timeout=3) for request in (1, 2)]
            assert [f.result(timeout=4) for f in futures] == [10, 20]
    finally:
        worker.stop()


def test_cpu_preparation_overlaps_inflight_gpu_work(worker_class):
    gpu_started, second_prepared, release_gpu = threading.Event(), threading.Event(), threading.Event()
    class OverlapPolicy(Policy):
        def prepare_request(self, request):
            if request == 2:
                second_prepared.set()
            return request * 10
        def predict_prepared_batch(self, requests):
            if requests == [10]:
                gpu_started.set()
                assert release_gpu.wait(timeout=3)
            return requests
    worker = worker_class(OverlapPolicy(), max_batch_size=1, batch_window_s=0)
    worker.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(worker.predict, 1, timeout=4)
            assert gpu_started.wait(timeout=2)
            second = pool.submit(worker.predict, 2, timeout=4)
            try:
                assert second_prepared.wait(timeout=2)
            finally:
                release_gpu.set()
            assert first.result(timeout=3) == 10
            assert second.result(timeout=3) == 20
    finally:
        release_gpu.set()
        worker.stop()


def test_preparation_error_does_not_poison_worker(worker_class):
    class BadInputPolicy(Policy):
        def prepare_request(self, request):
            if request < 0:
                raise ValueError("bad frame")
            return request * 10
    worker = worker_class(BadInputPolicy(), max_batch_size=1, batch_window_s=0)
    worker.start()
    try:
        with pytest.raises(ValueError, match="bad frame"):
            worker.predict(-1, timeout=1)
        assert worker.predict(3, timeout=1) == 30
    finally:
        worker.stop()


def test_stop_while_preparing_rejects_enqueue(worker_class):
    entered, resume = threading.Event(), threading.Event()
    class PausedPolicy(Policy):
        def prepare_request(self, request):
            entered.set()
            assert resume.wait(timeout=3)
            return request
    worker = worker_class(PausedPolicy(), max_batch_size=1, batch_window_s=0)
    worker.start()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker.predict, 1, timeout=4)
        assert entered.wait(timeout=2)
        worker.stop()
        resume.set()
        with pytest.raises(RuntimeError, match="not running|stopping"):
            future.result(timeout=2)


def test_timed_out_work_retains_capacity_until_gpu_finishes(worker_class):
    entered, release = threading.Event(), threading.Event()
    prepared = []
    class SlowPolicy(Policy):
        def prepare_request(self, request):
            prepared.append(request)
            return request
        def predict_prepared_batch(self, requests):
            entered.set()
            assert release.wait(timeout=3)
            return requests
    worker = worker_class(SlowPolicy(), max_batch_size=1, batch_window_s=0)
    worker.start()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(worker.predict, 1, timeout=0.2)
            assert entered.wait(timeout=2)
            with pytest.raises(TimeoutError):
                first.result(timeout=2)
        with pytest.raises(TimeoutError):
            worker.predict(2, timeout=0.05)
        with pytest.raises(TimeoutError, match="capacity"):
            worker.predict(3, timeout=0.05)
        assert prepared == [1, 2]
    finally:
        release.set()
        worker.stop()

