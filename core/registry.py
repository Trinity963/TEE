"""
TEE — Trinity's Execution Engine
core/registry.py

Model directory watcher and live manifest manager.
Watches all configured model directories.
Calls detector automatically when a GGUF drops.
Maintains the live manifest of every model TEE knows about.
No restart required — hot-drop works.

MIT License — open source, sovereign, forever.
"""

import json
import os
import time
import threading
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

from detector import Detector

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  TEE:registry  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("tee.registry")

# ── Supported model formats ───────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    ".gguf":        "gguf",
    ".ggml":        "ggml",
}

MODELFILE_SUFFIX = ".modelfile.json"

# ── Scan interval for watch loop (seconds) ───────────────────────────────────

WATCH_INTERVAL = 5


# ─────────────────────────────────────────────────────────────────────────────
# ModelEntry — one registered model
# ─────────────────────────────────────────────────────────────────────────────

class ModelEntry:
    """
    Represents one registered model in the TEE manifest.
    Built from a detected modelfile or a user-supplied one.
    """

    def __init__(self, modelfile: dict, modelfile_path: str, gguf_path: str):
        self.modelfile_path = Path(modelfile_path)
        self.gguf_path      = Path(gguf_path)
        self.data           = modelfile

        # Core identity
        self.name         = modelfile.get("name", "unknown")
        self.format       = modelfile.get("format", "gguf")
        self.architecture = modelfile.get("architecture", "unknown")
        self.parameters   = modelfile.get("parameters", "unknown")
        self.quantization = modelfile.get("quantization", "unknown")
        self.context      = modelfile.get("context", 4096)
        self.backend      = modelfile.get("backend", "auto")
        self.gpu          = modelfile.get("gpu", "auto")
        self.tags         = modelfile.get("tags", [])
        self.size_gb      = round(self.gguf_path.stat().st_size / (1024 ** 3), 2)

        # Runtime state
        self.status       = "registered"   # registered | loaded | unloading | error
        self.registered_at = datetime.now(timezone.utc).isoformat()
        self.loaded_at    = None
        self.error        = None

    def to_dict(self) -> dict:
        return {
            "name":          self.name,
            "format":        self.format,
            "architecture":  self.architecture,
            "parameters":    self.parameters,
            "quantization":  self.quantization,
            "context":       self.context,
            "backend":       self.backend,
            "gpu":           self.gpu,
            "tags":          self.tags,
            "size_gb":       self.size_gb,
            "status":        self.status,
            "registered_at": self.registered_at,
            "loaded_at":     self.loaded_at,
            "gguf_path":     str(self.gguf_path),
            "modelfile_path": str(self.modelfile_path),
            "error":         self.error,
        }

    def to_api_dict(self) -> dict:
        """OpenAI-compatible model list entry."""
        return {
            "id":       self.name,
            "object":   "model",
            "created":  int(time.time()),
            "owned_by": "tee",
            "permission": [],
            "root":     self.name,
            "parent":   None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Registry — the live manifest
# ─────────────────────────────────────────────────────────────────────────────

class Registry:
    """
    Maintains the live manifest of all models TEE knows about.
    Watches configured directories for new model files.
    Calls Detector automatically on new GGUFs.
    Thread-safe — runtime and gateway read this concurrently.
    """

    def __init__(self, config_path: str = None):
        self._models:   Dict[str, ModelEntry] = {}
        self._lock:     threading.RLock       = threading.RLock()
        self._watched:  List[dict]            = []
        self._seen:     Dict[str, float]      = {}   # path → mtime
        self._watcher:  Optional[threading.Thread] = None
        self._running:  bool                  = False
        self._config:   dict                  = {}

        if config_path:
            self._load_config(config_path)

    # ── Config ────────────────────────────────────────────────────────────────

    def _load_config(self, config_path: str):
        """Load tee.config and extract model directory settings."""
        path = Path(config_path)
        if not path.exists():
            log.warning(f"Config not found: {config_path} — using defaults")
            return

        with open(path) as f:
            self._config = json.load(f)

        dirs = self._config.get("models_dirs", [])
        for entry in dirs:
            self.add_watch_dir(
                path=entry.get("path", ""),
                label=entry.get("label", "auto"),
                watch=entry.get("watch", True),
            )

    def add_watch_dir(self, path: str, label: str = "auto", watch: bool = True):
        """Add a directory to the watch list."""
        p = Path(path)
        if not p.exists():
            log.warning(f"Models dir not found: {path} — skipping")
            return

        entry = {"path": str(p), "label": label, "watch": watch}
        self._watched.append(entry)
        log.info(f"Watching: {p}  [{label}]")

    # ── Scan ──────────────────────────────────────────────────────────────────

    def scan_all(self):
        """Scan all watched directories and register any models found."""
        for entry in self._watched:
            self._scan_dir(entry["path"])

    def _scan_dir(self, dir_path: str):
        """Scan one directory for model files."""
        p = Path(dir_path)
        if not p.exists():
            return

        for file in p.iterdir():
            if file.suffix.lower() in SUPPORTED_EXTENSIONS:
                self._process_model_file(file)

    def _process_model_file(self, gguf_path: Path):
        """
        Process a discovered GGUF file.
        Uses existing modelfile if present, generates one if not.
        Skips if already registered and unchanged.
        """
        try:
            mtime = gguf_path.stat().st_mtime
        except OSError:
            return

        key = str(gguf_path)

        # Skip if already seen and unchanged
        if key in self._seen and self._seen[key] == mtime:
            return

        self._seen[key] = mtime

        # Find or generate modelfile
        modelfile_path = gguf_path.with_suffix(MODELFILE_SUFFIX)

        if modelfile_path.exists():
            log.info(f"Found modelfile: {modelfile_path.name}")
            try:
                with open(modelfile_path) as f:
                    modelfile = json.load(f)
            except Exception as e:
                log.error(f"Failed to read modelfile {modelfile_path}: {e}")
                return
        else:
            log.info(f"New model detected: {gguf_path.name} — auto-generating modelfile")
            try:
                d             = Detector(str(gguf_path))
                info          = d.detect()
                written_path  = d.generate_modelfile(str(modelfile_path))
                with open(written_path) as f:
                    modelfile = json.load(f)
                log.info(f"Modelfile generated: {modelfile_path.name}")
            except Exception as e:
                log.error(f"Detector failed on {gguf_path.name}: {e}")
                return

        # Register
        self._register(modelfile, str(modelfile_path), str(gguf_path))

    def _register(self, modelfile: dict, modelfile_path: str, gguf_path: str):
        """Add or update a model entry in the manifest."""
        try:
            entry = ModelEntry(modelfile, modelfile_path, gguf_path)
        except Exception as e:
            log.error(f"Failed to create ModelEntry: {e}")
            return

        with self._lock:
            existing = self._models.get(entry.name)
            if existing:
                # Preserve runtime state on re-registration
                entry.status    = existing.status
                entry.loaded_at = existing.loaded_at

            self._models[entry.name] = entry

        log.info(
            f"✓ Registered: {entry.name}  "
            f"[{entry.architecture} {entry.parameters} {entry.quantization}  "
            f"{entry.size_gb}GB  ctx:{entry.context:,}]"
        )

    # ── Watcher ───────────────────────────────────────────────────────────────

    def start_watching(self):
        """Start the background directory watcher thread."""
        if self._running:
            return

        self._running = True
        self._watcher = threading.Thread(
            target=self._watch_loop,
            name="tee-registry-watcher",
            daemon=True,
        )
        self._watcher.start()
        log.info("Directory watcher started")

    def stop_watching(self):
        """Stop the background watcher thread."""
        self._running = False
        if self._watcher:
            self._watcher.join(timeout=10)
        log.info("Directory watcher stopped")

    def _watch_loop(self):
        """Background loop — scans all dirs every WATCH_INTERVAL seconds."""
        while self._running:
            try:
                self.scan_all()
            except Exception as e:
                log.error(f"Watch loop error: {e}")
            time.sleep(WATCH_INTERVAL)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_model(self, name: str) -> Optional[ModelEntry]:
        """Return a model entry by name. None if not found."""
        with self._lock:
            return self._models.get(name)

    def list_models(self) -> List[ModelEntry]:
        """Return all registered models."""
        with self._lock:
            return list(self._models.values())

    def list_models_api(self) -> dict:
        """Return OpenAI-compatible model list."""
        with self._lock:
            return {
                "object": "list",
                "data":   [m.to_api_dict() for m in self._models.values()],
            }

    def get_manifest(self) -> dict:
        """Return the full manifest — all models with all details."""
        with self._lock:
            return {
                "tee_version":  "1.0.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model_count":  len(self._models),
                "models":       {k: v.to_dict() for k, v in self._models.items()},
            }

    def update_status(self, name: str, status: str, error: str = None):
        """Update a model's runtime status. Called by runtime.py."""
        with self._lock:
            entry = self._models.get(name)
            if entry:
                entry.status = status
                entry.error  = error
                if status == "loaded":
                    entry.loaded_at = datetime.now(timezone.utc).isoformat()

    def remove_model(self, name: str):
        """Remove a model from the registry. Called when file is deleted."""
        with self._lock:
            if name in self._models:
                del self._models[name]
                log.info(f"Unregistered: {name}")

    def model_count(self) -> int:
        with self._lock:
            return len(self._models)

    def print_manifest(self):
        """Print a human-readable manifest summary."""
        with self._lock:
            models = list(self._models.values())

        if not models:
            print("\nTEE Registry — no models registered")
            return

        print(f"\nTEE Registry — {len(models)} model(s) registered")
        print("─" * 70)
        for m in models:
            print(f"  {m.name}")
            print(f"    {m.architecture} · {m.parameters} · {m.quantization} · {m.size_gb}GB · ctx:{m.context:,}")
            print(f"    backend:{m.backend}  gpu:{m.gpu}  status:{m.status}")
            print(f"    tags: {', '.join(m.tags)}")
            print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI — python3 registry.py /path/to/models [/path/to/more/models]
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 registry.py /path/to/models [/path/to/more]")
        sys.exit(1)

    reg = Registry()

    for d in sys.argv[1:]:
        reg.add_watch_dir(d, label="auto", watch=True)

    print("\nTEE Registry — scanning...")
    reg.scan_all()
    reg.print_manifest()

    print("Watching for new models — Ctrl+C to stop\n")
    reg.start_watching()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        reg.stop_watching()
        print("\nRegistry stopped.")
