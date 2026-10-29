from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKER_PYTHON_ENV = "IMAGE_EDIT_WORKER_PYTHON"
EXTERNAL_WORKER_BACKENDS = {"external_worker", "worker", "subprocess_worker"}

_WORKERS: dict[tuple[str, str, str], "ExternalImageEditWorker"] = {}
_WORKERS_LOCK = threading.Lock()


@dataclass(frozen=True)
class WorkerConfig:
    python: str
    script: str
    cwd: str
    timeout: float


def resolve_worker_config(config: dict[str, Any]) -> WorkerConfig:
    project_dir = Path(str(config.get("_project_dir") or Path(__file__).resolve().parents[1])).expanduser()
    worker_python = str(config.get("worker_python") or os.getenv(WORKER_PYTHON_ENV) or "").strip()
    if not worker_python:
        raise ValueError(
            "external image edit worker requires image_edit.worker_python or "
            f"the {WORKER_PYTHON_ENV} environment variable"
        )
    script_value = str(config.get("worker_script") or Path(__file__).with_name("image_edit_worker.py"))
    script = Path(script_value).expanduser()
    if not script.is_absolute():
        script = project_dir / script
    cwd_value = str(config.get("worker_cwd") or project_dir)
    cwd = Path(cwd_value).expanduser()
    if not cwd.is_absolute():
        cwd = project_dir / cwd
    return WorkerConfig(
        python=worker_python,
        script=str(script),
        cwd=str(cwd),
        timeout=float(config.get("worker_timeout", config.get("request_timeout", 600)) or 600),
    )


def request_external_image_edit(prompt: str, image_url: str, config: dict[str, Any]) -> dict[str, Any]:
    worker_config = resolve_worker_config(config)
    key = (worker_config.python, worker_config.script, worker_config.cwd)
    with _WORKERS_LOCK:
        worker = _WORKERS.get(key)
        if worker is None or not worker.is_alive():
            worker = ExternalImageEditWorker(worker_config)
            _WORKERS[key] = worker
    return worker.request(prompt, image_url, config)


class ExternalImageEditWorker:
    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def request(self, prompt: str, image_url: str, config: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            process = self._ensure_process()
            request_id = uuid.uuid4().hex
            payload = {
                "request_id": request_id,
                "prompt": str(prompt),
                "image_url": str(image_url),
                "config": dict(config),
            }
            try:
                assert process.stdin is not None
                process.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                process.stdin.flush()
            except BrokenPipeError as exc:
                self._process = None
                raise RuntimeError("image edit worker exited before accepting request") from exc
            return self._read_response(process, request_id)

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        self._process = subprocess.Popen(
            [self.config.python, self.config.script],
            cwd=self.config.cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        return self._process

    def _read_response(self, process: subprocess.Popen[str], request_id: str) -> dict[str, Any]:
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + self.config.timeout
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    self._process = None
                    raise RuntimeError(f"image edit worker exited with code {process.returncode}")
                events = selector.select(timeout=0.2)
                if not events:
                    continue
                line = process.stdout.readline()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    print(f"image edit worker stdout noise: {line.rstrip()}", file=sys.stderr)
                    continue
                if str(message.get("request_id") or "") != request_id:
                    print(f"image edit worker ignored stale response: {message!r}", file=sys.stderr)
                    continue
                if not bool(message.get("ok")):
                    raise RuntimeError(str(message.get("error") or "image edit worker request failed"))
                value = message.get("response")
                if not isinstance(value, dict):
                    raise RuntimeError(f"image edit worker returned non-object response: {value!r}")
                return value
            raise TimeoutError(f"image edit worker timed out after {self.config.timeout:g}s")
        finally:
            selector.close()
