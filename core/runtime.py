"""
TEE — Trinity's Execution Engine
core/runtime.py

GPU placement, backend selection, model load and unload.
Reads the registry. Decides which backend runs which model on which GPU.
Handles concurrent requests. Manages VRAM budget across all GPUs.
No cloud. No external dependencies. Sovereign inference.

MIT License — open source, sovereign, forever.
"""

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from registry import Registry, ModelEntry

# ── Logging ───────────────────────────────────────────────────────────────────

log = logging.getLogger("tee.runtime")

# ── Backend availability ──────────────────────────────────────────────────────

BACKENDS = {
    "llama.cpp": {
        "formats":    ["gguf", "ggml"],
        "server_bin": "llama-server",
        "alt_bins":   ["llama.cpp/llama-server", "./llama-server", "/srv/Build_Core/llama.cpp/build/bin/llama-server"],
    },
    "vllm": {
        "formats":    ["safetensors"],
        "server_bin": "vllm",
        "alt_bins":   [],
    },
}

# ── VRAM safety margin — never fill a GPU completely ─────────────────────────

VRAM_SAFETY_MARGIN_GB = 1.5

# ── Idle unload default (minutes) ────────────────────────────────────────────

IDLE_UNLOAD_MINUTES = 30

# ── Port base for backend servers ────────────────────────────────────────────

PORT_BASE = 9100


# ─────────────────────────────────────────────────────────────────────────────
# GPUInfo — detected GPU state
# ─────────────────────────────────────────────────────────────────────────────

class GPUInfo:
    """Represents one physical GPU detected on the system."""

    def __init__(self, index: int, name: str, vram_total_gb: float):
        self.index        = index
        self.name         = name
        self.vram_total_gb = vram_total_gb
        self.vram_used_gb  = 0.0
        self.models:  List[str] = []   # model names loaded on this GPU

    @property
    def vram_free_gb(self) -> float:
        return round(self.vram_total_gb - self.vram_used_gb - VRAM_SAFETY_MARGIN_GB, 2)

    @property
    def vram_free_pct(self) -> float:
        if self.vram_total_gb == 0:
            return 0.0
        return round((self.vram_free_gb / self.vram_total_gb) * 100, 1)

    def can_fit(self, size_gb: float) -> bool:
        return self.vram_free_gb >= size_gb

    def to_dict(self) -> dict:
        return {
            "index":          self.index,
            "name":           self.name,
            "vram_total_gb":  self.vram_total_gb,
            "vram_used_gb":   round(self.vram_used_gb, 2),
            "vram_free_gb":   self.vram_free_gb,
            "vram_free_pct":  self.vram_free_pct,
            "models_loaded":  self.models,
        }


# ─────────────────────────────────────────────────────────────────────────────
# LoadedModel — a model currently running in a backend process
# ─────────────────────────────────────────────────────────────────────────────

class LoadedModel:
    """Tracks a model that is currently loaded and serving inference."""

    def __init__(
        self,
        name:       str,
        backend:    str,
        gpu_ids:    List[int],
        port:       int,
        process:    Optional[subprocess.Popen],
        size_gb:    float,
    ):
        self.name       = name
        self.backend    = backend
        self.gpu_ids    = gpu_ids
        self.port       = port
        self.process    = process
        self.size_gb    = size_gb
        self.loaded_at  = time.time()
        self.last_used  = time.time()
        self.request_count = 0

    def touch(self):
        """Update last used time on every request."""
        self.last_used     = time.time()
        self.request_count += 1

    def idle_minutes(self) -> float:
        return (time.time() - self.last_used) / 60.0

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def to_dict(self) -> dict:
        return {
            "name":          self.name,
            "backend":       self.backend,
            "gpu_ids":       self.gpu_ids,
            "port":          self.port,
            "size_gb":       self.size_gb,
            "loaded_at":     self.loaded_at,
            "last_used":     self.last_used,
            "idle_minutes":  round(self.idle_minutes(), 1),
            "request_count": self.request_count,
            "base_url":      self.base_url(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Runtime — the sovereign inference engine
# ─────────────────────────────────────────────────────────────────────────────

class Runtime:
    """
    Sovereign inference runtime.
    Detects GPUs. Selects backends. Places models. Manages VRAM.
    Handles load, unload, and idle cleanup automatically.
    """

    def __init__(self, registry: Registry, config: dict = None):
        self._registry  = registry
        self._config    = config or {}
        self._gpus:     Dict[int, GPUInfo]    = {}
        self._loaded:   Dict[str, LoadedModel] = {}
        self._lock      = threading.RLock()
        self._port_counter = PORT_BASE
        self._idle_thread: Optional[threading.Thread] = None
        self._running   = False

        self._idle_unload_minutes = self._config.get(
            "runtime", {}
        ).get("unload_idle_after_minutes", IDLE_UNLOAD_MINUTES)

    # ── Startup ───────────────────────────────────────────────────────────────

    def start(self):
        """Detect hardware and start the idle monitor."""
        log.info("TEE Runtime starting...")
        self._detect_gpus()
        self._detect_backends()
        self._start_idle_monitor()
        self._running = True
        log.info(f"Runtime ready — {len(self._gpus)} GPU(s) available")

    def stop(self):
        """Unload all models and stop the runtime."""
        self._running = False
        log.info("Runtime shutting down — unloading all models...")
        with self._lock:
            for name in list(self._loaded.keys()):
                self._unload(name)
        log.info("Runtime stopped.")

    # ── GPU Detection ─────────────────────────────────────────────────────────

    def _detect_gpus(self):
        """Detect available NVIDIA GPUs via nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError("nvidia-smi failed")

            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                idx       = int(parts[0])
                name      = parts[1]
                vram_mb   = float(parts[2])
                vram_gb   = round(vram_mb / 1024, 2)
                gpu       = GPUInfo(idx, name, vram_gb)
                self._gpus[idx] = gpu
                log.info(f"GPU {idx}: {name}  {vram_gb}GB VRAM")

        except FileNotFoundError:
            log.warning("nvidia-smi not found — CPU-only mode")
        except Exception as e:
            log.warning(f"GPU detection failed: {e} — CPU-only mode")

        if not self._gpus:
            log.info("Running in CPU-only mode")

    def _detect_backends(self):
        """Check which backends are available on this system."""
        for name, info in BACKENDS.items():
            found = self._find_binary(info["server_bin"], info["alt_bins"])
            if found:
                log.info(f"Backend available: {name}  [{found}]")
            else:
                log.warning(f"Backend not found: {name} — install to enable")

    def _find_binary(self, name: str, alternatives: List[str]) -> Optional[str]:
        """Find a backend binary in PATH or known locations."""
        import shutil
        if shutil.which(name):
            return name
        for alt in alternatives:
            if Path(alt).exists():
                return alt
        return None

    # ── Placement ─────────────────────────────────────────────────────────────

    def _select_backend(self, entry: ModelEntry) -> str:
        """Select the best backend for a model format."""
        if entry.backend != "auto":
            return entry.backend
        for backend, info in BACKENDS.items():
            if entry.format in info["formats"]:
                if self._find_binary(info["server_bin"], info["alt_bins"]):
                    return backend
        return "llama.cpp"   # safe default

    def _select_gpus(self, size_gb: float) -> Tuple[List[int], str]:
        """
        Select which GPU(s) to run a model on.
        Returns (gpu_id_list, placement_label).

        Logic:
          1. Single GPU — fits comfortably     → use it
          2. Single GPU — fits but tight        → use it with warning
          3. Multi-GPU  — span two GPUs         → split across them
          4. No GPU / no fit                    → CPU mode
        """
        if not self._gpus:
            return [], "cpu"

        # Try single GPU — prefer most free VRAM
        candidates = sorted(
            self._gpus.values(),
            key=lambda g: g.vram_free_gb,
            reverse=True,
        )

        for gpu in candidates:
            if gpu.can_fit(size_gb):
                label = "single-gpu"
                if gpu.vram_free_gb - size_gb < 2.0:
                    label = "single-gpu-tight"
                return [gpu.index], label

        # Try multi-GPU span
        if len(self._gpus) >= 2:
            total_free = sum(g.vram_free_gb for g in self._gpus.values())
            if total_free >= size_gb:
                gpu_ids = [g.index for g in candidates]
                return gpu_ids, "multi-gpu"

        # CPU fallback
        log.warning(
            f"Model size {size_gb}GB exceeds available VRAM — falling back to CPU"
        )
        return [], "cpu"

    def _compatibility_label(self, size_gb: float) -> dict:
        """
        Return a plain-English compatibility assessment.
        Used by the gateway to inform users before download.
        """
        if not self._gpus:
            # CPU only
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            if size_gb <= ram_gb * 0.6:
                return {"status": "cpu_ok",      "label": "✅ Runs on CPU alone"}
            return     {"status": "cpu_no_ram",  "label": "🚫 Cannot Run — insufficient RAM"}

        gpu_ids, placement = self._select_gpus(size_gb)

        if placement == "single-gpu":
            return {"status": "single_gpu",      "label": "✅ Runs on Single GPU"}
        if placement == "single-gpu-tight":
            return {"status": "single_gpu_tight","label": "⚠️  Runs on Single GPU — will be tight"}
        if placement == "multi-gpu":
            return {"status": "multi_gpu",       "label": "🔴 Needs Multi-GPU"}
        return         {"status": "cannot_run",  "label": "🚫 Cannot Run — exceeds all available VRAM"}

    # ── Load / Unload ─────────────────────────────────────────────────────────

    def load(self, name: str) -> Optional[LoadedModel]:
        """
        Load a model into a backend server.
        Returns the LoadedModel if successful.
        """
        with self._lock:
            # Already loaded
            if name in self._loaded:
                self._loaded[name].touch()
                return self._loaded[name]

            entry = self._registry.get_model(name)
            if not entry:
                log.error(f"Model not found in registry: {name}")
                return None

            backend   = self._select_backend(entry)
            gpu_ids, placement = self._select_gpus(entry.size_gb)
            port      = self._next_port()

            log.info(
                f"Loading: {name}  backend:{backend}  "
                f"placement:{placement}  port:{port}"
            )

            # Ollama models are already running — no process to spawn
            if backend == "ollama":
                ollama_port = int(entry.ollama_base.split(":")[-1]) if hasattr(entry, "ollama_base") else 11434
                loaded = LoadedModel(
                    name    = name,
                    backend = "ollama",
                    gpu_ids = [],
                    port    = ollama_port,
                    process = None,
                    size_gb = entry.size_gb,
                )
                self._loaded[name] = loaded
                self._registry.update_status(name, "loaded")
                log.info(f"✓ Ollama passthrough: {name}  → {loaded.base_url()}")
                return loaded

            process = self._launch_backend(entry, backend, gpu_ids, port)
            if process is None:
                self._registry.update_status(name, "error", "Backend failed to launch")
                return None

            loaded = LoadedModel(
                name    = name,
                backend = backend,
                gpu_ids = gpu_ids,
                port    = port,
                process = process,
                size_gb = entry.size_gb,
            )
            self._loaded[name] = loaded

            # Update VRAM accounting
            for gid in gpu_ids:
                if gid in self._gpus:
                    self._gpus[gid].vram_used_gb += entry.size_gb / max(len(gpu_ids), 1)
                    self._gpus[gid].models.append(name)

            self._registry.update_status(name, "loaded")
            log.info(f"✓ Loaded: {name}  → {loaded.base_url()}")
            return loaded

    def _launch_backend(
        self,
        entry:   ModelEntry,
        backend: str,
        gpu_ids: List[int],
        port:    int,
    ) -> Optional[subprocess.Popen]:
        """Launch the backend server process for a model."""

        if backend == "llama.cpp":
            return self._launch_llamacpp(entry, gpu_ids, port)
        elif backend == "vllm":
            return self._launch_vllm(entry, gpu_ids, port)
        else:
            log.error(f"Unknown backend: {backend}")
            return None

    def _launch_llamacpp(
        self,
        entry:   ModelEntry,
        gpu_ids: List[int],
        port:    int,
    ) -> Optional[subprocess.Popen]:
        """Launch llama-server for a GGUF model."""
        binary = self._find_binary("llama-server", BACKENDS["llama.cpp"]["alt_bins"])
        if not binary:
            log.error("llama-server not found — cannot load GGUF model")
            return None

        n_gpu_layers = -1 if gpu_ids else 0   # -1 = all layers on GPU

        cmd = [
            binary,
            "--model",    str(entry.gguf_path),
            "--port",     str(port),
            "--host",     "127.0.0.1",
            "--ctx-size", str(entry.context),
            "--n-gpu-layers", str(n_gpu_layers),
            "--threads",  str(os.cpu_count() or 4),
            "--silent-prompt",
        ]

        if gpu_ids:
            cmd += ["--main-gpu", str(gpu_ids[0])]
            if len(gpu_ids) > 1:
                cmd += ["--tensor-split", ",".join(["1"] * len(gpu_ids))]

        env = os.environ.copy()
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            # Give the server a moment to start
            time.sleep(2)
            if process.poll() is not None:
                log.error(f"llama-server exited immediately for {entry.name}")
                return None
            return process
        except Exception as e:
            log.error(f"Failed to launch llama-server: {e}")
            return None

    def _launch_vllm(
        self,
        entry:   ModelEntry,
        gpu_ids: List[int],
        port:    int,
    ) -> Optional[subprocess.Popen]:
        """Launch vLLM for a safetensors model."""
        binary = self._find_binary("vllm", BACKENDS["vllm"]["alt_bins"])
        if not binary:
            log.error("vllm not found — cannot load safetensors model")
            return None

        cmd = [
            binary, "serve",
            str(entry.gguf_path),
            "--port",  str(port),
            "--host",  "127.0.0.1",
        ]

        if len(gpu_ids) > 1:
            cmd += ["--tensor-parallel-size", str(len(gpu_ids))]

        env = os.environ.copy()
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            time.sleep(3)
            if process.poll() is not None:
                log.error(f"vllm exited immediately for {entry.name}")
                return None
            return process
        except Exception as e:
            log.error(f"Failed to launch vllm: {e}")
            return None

    def _unload(self, name: str):
        """Unload a model — terminate its backend process."""
        loaded = self._loaded.get(name)
        if not loaded:
            return

        log.info(f"Unloading: {name}")

        if loaded.process and loaded.process.poll() is None:
            loaded.process.terminate()
            try:
                loaded.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                loaded.process.kill()

        # Free VRAM accounting
        for gid in loaded.gpu_ids:
            if gid in self._gpus:
                self._gpus[gid].vram_used_gb = max(
                    0.0,
                    self._gpus[gid].vram_used_gb - loaded.size_gb / max(len(loaded.gpu_ids), 1)
                )
                if name in self._gpus[gid].models:
                    self._gpus[gid].models.remove(name)

        del self._loaded[name]
        self._registry.update_status(name, "registered")
        log.info(f"✓ Unloaded: {name}")

    # ── Idle monitor ──────────────────────────────────────────────────────────

    def _start_idle_monitor(self):
        """Start the background idle unload monitor."""
        self._idle_thread = threading.Thread(
            target=self._idle_loop,
            name="tee-idle-monitor",
            daemon=True,
        )
        self._idle_thread.start()
        log.info(
            f"Idle monitor started — unload after {self._idle_unload_minutes}min"
        )

    def _idle_loop(self):
        """Background loop — unloads models idle past the threshold."""
        while self._running:
            try:
                with self._lock:
                    idle = [
                        name for name, m in self._loaded.items()
                        if m.idle_minutes() >= self._idle_unload_minutes
                    ]
                for name in idle:
                    log.info(
                        f"Idle unload: {name} "
                        f"({self._loaded[name].idle_minutes():.1f}min idle)"
                    )
                    with self._lock:
                        self._unload(name)
            except Exception as e:
                log.error(f"Idle monitor error: {e}")
            time.sleep(60)

    # ── Port management ───────────────────────────────────────────────────────

    def _next_port(self) -> int:
        """Return the next available port for a backend server."""
        used = {m.port for m in self._loaded.values()}
        while self._port_counter in used:
            self._port_counter += 1
        port = self._port_counter
        self._port_counter += 1
        return port

    # ── Public API ────────────────────────────────────────────────────────────

    def get_loaded(self, name: str) -> Optional[LoadedModel]:
        """Return a loaded model by name. None if not loaded."""
        with self._lock:
            return self._loaded.get(name)

    def get_or_load(self, name: str) -> Optional[LoadedModel]:
        """Return loaded model — loading it first if needed."""
        with self._lock:
            if name in self._loaded:
                self._loaded[name].touch()
                return self._loaded[name]
        return self.load(name)

    def list_loaded(self) -> List[LoadedModel]:
        """Return all currently loaded models."""
        with self._lock:
            return list(self._loaded.values())

    def gpu_status(self) -> List[dict]:
        """Return current GPU status."""
        with self._lock:
            return [g.to_dict() for g in self._gpus.values()]

    def system_status(self) -> dict:
        """Return full system status — GPUs, loaded models, registry count."""
        with self._lock:
            return {
                "gpus":           [g.to_dict() for g in self._gpus.values()],
                "loaded_models":  [m.to_dict() for m in self._loaded.values()],
                "registered_models": self._registry.model_count(),
                "cpu_only":       len(self._gpus) == 0,
            }

    def print_status(self):
        """Print a human-readable system status."""
        status = self.system_status()
        print("\nTEE Runtime Status")
        print("─" * 50)

        if status["cpu_only"]:
            print("  Mode: CPU only")
        else:
            for gpu in status["gpus"]:
                bar_filled = int((1 - gpu["vram_free_pct"] / 100) * 20)
                bar = "█" * bar_filled + "░" * (20 - bar_filled)
                print(
                    f"  GPU {gpu['index']}: {gpu['name']}\n"
                    f"    [{bar}] {gpu['vram_free_gb']}GB free / {gpu['vram_total_gb']}GB\n"
                    f"    Loaded: {', '.join(gpu['models_loaded']) or 'none'}"
                )

        print(f"\n  Registered models: {status['registered_models']}")
        print(f"  Loaded models:     {len(status['loaded_models'])}")
        for m in status["loaded_models"]:
            print(f"    · {m['name']}  {m['backend']}  GPU:{m['gpu_ids']}  {m['base_url']}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI — python3 runtime.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Quick system check — no models loaded, just detect and report
    print("\nTEE Runtime — system check")
    print("─" * 50)

    reg = Registry()
    rt  = Runtime(reg)
    rt._detect_gpus()
    rt._detect_backends()
    rt.print_status()
