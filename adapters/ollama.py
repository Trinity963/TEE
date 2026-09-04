#!/usr/bin/env python3
"""
TEE — Trinity's Execution Engine
adapters/ollama.py
Ollama adapter — discovers and registers models served by a local Ollama instance.
Runs alongside the GGUF registry. Ollama models are routed to localhost:11434.
Skips cloud/remote models (size < 1MB).
MIT License — open source, sovereign, forever.
"""
import json
import logging
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("tee.ollama")

OLLAMA_API      = "http://127.0.0.1:11434"
POLL_INTERVAL   = 30   # seconds — how often to re-check Ollama for new models
MIN_SIZE_BYTES  = 1_000_000  # skip cloud stubs (size < 1MB)


# ─────────────────────────────────────────────────────────────────────────────
# OllamaModelEntry — ModelEntry-compatible, no gguf_path stat needed
# ─────────────────────────────────────────────────────────────────────────────
class OllamaModelEntry:
    """
    Registry entry for an Ollama-managed model.
    Interface-compatible with ModelEntry so registry and gateway treat it identically.
    """
    def __init__(self, m: dict):
        details = m.get("details", {})
        self.name         = m["name"]
        self.format       = details.get("format", "gguf")
        self.architecture = details.get("family", "unknown")
        self.parameters   = details.get("parameter_size", "unknown")
        self.quantization = details.get("quantization_level", "unknown")
        self.context      = details.get("context_length", 4096)
        self.backend      = "ollama"
        self.gpu          = "auto"
        self.tags         = self._infer_tags(m)
        self.size_gb      = round(m.get("size", 0) / (1024 ** 3), 2)
        # Paths — not applicable for Ollama models
        self.modelfile_path = Path("/dev/null")
        self.gguf_path      = Path("/dev/null")
        self.data           = {}
        # Runtime state
        self.status        = "registered"
        self.registered_at = datetime.now(timezone.utc).isoformat()
        self.loaded_at     = None
        self.error         = None
        # Ollama-specific
        self.ollama_model  = m.get("model", self.name)
        self.ollama_base   = OLLAMA_API

    def _infer_tags(self, m: dict) -> list:
        tags  = ["ollama"]
        name  = m.get("name", "").lower()
        caps  = m.get("capabilities", [])
        if "tools" in caps:
            tags.append("tools")
        if "thinking" in caps:
            tags.append("thinking")
        if "vision" in caps:
            tags.append("vision")
        if any(w in name for w in ["coder", "code"]):
            tags.append("code")
        if any(w in name for w in ["instruct", "chat"]):
            tags.append("instruct")
        return tags

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
            "gguf_path":     "ollama",
            "modelfile_path": "ollama",
            "error":         self.error,
            "ollama_model":  self.ollama_model,
            "ollama_base":   self.ollama_base,
        }

    def to_api_dict(self) -> dict:
        return {
            "id":       self.name,
            "object":   "model",
            "created":  int(time.time()),
            "owned_by": "ollama",
            "permission": [],
            "root":     self.name,
            "parent":   None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# OllamaAdapter — polls Ollama, registers models into TEE registry
# ─────────────────────────────────────────────────────────────────────────────
class OllamaAdapter:
    """
    Polls the local Ollama API and injects discovered models into the TEE registry.
    Runs as a background thread. Removes models that disappear from Ollama.
    """
    def __init__(self, registry):
        self._registry  = registry
        self._running   = False
        self._thread    = None
        self._known     = set()   # model names currently registered via this adapter

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="tee-ollama-adapter",
            daemon=True,
        )
        self._thread.start()
        log.info(f"Ollama adapter started — polling {OLLAMA_API}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _fetch_models(self) -> list:
        try:
            req = urllib.request.urlopen(f"{OLLAMA_API}/api/tags", timeout=5)
            data = json.loads(req.read())
            models = data.get("models", [])
            # Filter out cloud stubs
            local = [m for m in models if m.get("size", 0) >= MIN_SIZE_BYTES]
            return local
        except Exception as e:
            log.warning(f"Ollama unreachable: {e}")
            return []

    def _poll_loop(self):
        # Initial poll immediately
        self._sync()
        while self._running:
            time.sleep(POLL_INTERVAL)
            self._sync()

    def _sync(self):
        models = self._fetch_models()
        if not models:
            return

        current_names = set()
        for m in models:
            entry = OllamaModelEntry(m)
            current_names.add(entry.name)
            # Only register if not already known to avoid log spam
            if entry.name not in self._known:
                self._inject(entry)
                self._known.add(entry.name)

        # Remove models that disappeared from Ollama
        gone = self._known - current_names
        for name in gone:
            self._registry.remove_model(name)
            self._known.discard(name)
            log.info(f"Ollama model removed: {name}")

    def _inject(self, entry: OllamaModelEntry):
        """Inject an OllamaModelEntry directly into the registry's model dict."""
        with self._registry._lock:
            existing = self._registry._models.get(entry.name)
            if existing:
                entry.status    = existing.status
                entry.loaded_at = existing.loaded_at
            self._registry._models[entry.name] = entry
        log.info(
            f"✓ Ollama: {entry.name}  "
            f"[{entry.architecture} {entry.parameters} {entry.quantization}  "
            f"{entry.size_gb}GB  ctx:{entry.context:,}]"
        )
