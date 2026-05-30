import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator


class _LocalInferenceGuard:
    def __init__(self):
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._holder_label = ""
        self._holder_since = 0.0

    def _set_holder(self, label: str):
        with self._state_lock:
            self._holder_label = label
            self._holder_since = time.perf_counter()

    def _clear_holder(self):
        with self._state_lock:
            self._holder_label = ""
            self._holder_since = 0.0

    def describe_holder(self) -> str:
        with self._state_lock:
            label = self._holder_label
            since = self._holder_since
        if not label:
            return ""
        held_for = max(0.0, time.perf_counter() - since)
        return f"{label}，已持有 {held_for:.2f}s"

    @contextmanager
    def acquire(
        self,
        task_label: str,
        log_callback: Callable[[str], None] | None = None,
        probe_interval_sec: float = 2.0,
    ) -> Iterator[None]:
        label = str(task_label or "本地推理任务").strip() or "本地推理任务"
        probe_interval = max(0.5, float(probe_interval_sec))
        waited = 0.0
        announced_wait = False

        while True:
            if self._lock.acquire(timeout=probe_interval):
                self._set_holder(label)
                try:
                    if announced_wait and callable(log_callback):
                        log_callback(f"{label} 已获取本地推理锁，累计等待 {waited:.2f}s")
                    yield
                finally:
                    self._clear_holder()
                    self._lock.release()
                return

            waited += probe_interval
            announced_wait = True
            if callable(log_callback):
                holder_desc = self.describe_holder()
                if holder_desc:
                    log_callback(
                        f"{label} 等待本地推理锁中，已等待 {waited:.2f}s；当前占用方: {holder_desc}"
                    )
                else:
                    log_callback(f"{label} 等待本地推理锁中，已等待 {waited:.2f}s")


_LOCAL_ONNX_INFERENCE_GUARD = _LocalInferenceGuard()


@contextmanager
def acquire_local_onnx_inference_lock(
    task_label: str,
    log_callback: Callable[[str], None] | None = None,
    probe_interval_sec: float = 2.0,
) -> Iterator[None]:
    with _LOCAL_ONNX_INFERENCE_GUARD.acquire(
        task_label=task_label,
        log_callback=log_callback,
        probe_interval_sec=probe_interval_sec,
    ):
        yield


def describe_local_onnx_inference_holder() -> str:
    return _LOCAL_ONNX_INFERENCE_GUARD.describe_holder()
