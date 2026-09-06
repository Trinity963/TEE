"""
TEE — Trinity's Execution Engine
adapters/openrouter.py

OpenRouter adapter — discovers free-tier models and registers them into TEE.
Routes inference requests through the OpenRouter API.
MIT License — open source, sovereign, forever.
"""

import json
import logging
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("tee.openrouter")

OPENROUTER_API  = "https://openrouter.ai/api/v1"
POLL_INTERVAL   = 300   # seconds — re-check every 5 minutes


# ─────────────────────────────────────────────────────────────────────────────
# OpenRouterModelEntry
# ─────────────────────────────────────────────────────────────────────────────

class OpenRouterModelEntry:
    """
    Registry entry for an OpenRouter-hosted model.
    Interface-compatible with ModelEntry and OllamaModelEntry.
    """
    def __init__(self, m: dict, api_key: str):
        self.name          = m["id"]
        self.format        = "openrouter"
        self.architecture  = m.get("architecture", {}).get("modality", "text")
        self.parameters    = "unknown"
        self.quantization  = "cloud"
        self.context       = m.get("context_length", 4096)
        self.backend       = "openrouter"
        self.gpu           = "remote"
        self.tags          = self._infer_tags(m)
        self.size_gb       = 0.0   # remote — no local VRAM cost
        self.modelfile_path = None
        self.gguf_path      = None
        self.data           = {}
        self.status         = "registered"
        self.registered_at  = datetime.now(timezone.utc).isoformat()
        self.loaded_at      = None
        self.error          = None
        # OpenRouter-specific
        self.or_model_id   = m["id"]
        self.or_base       = OPENROUTER_API
        self.or_api_key    = api_key
        self.or_pricing    = m.get("pricing", {})
        self.or_name       = m.get("name", m["id"])

    def _infer_tags(self, m: dict) -> list:
        tags = ["openrouter"]
        name = m.get("id", "").lower()
        pricing = m.get("pricing", {})
        if str(pricing.get("completion", "1")) == "0":
            tags.append("free")
        if any(w in name for w in ["code", "coder", "coding"]):
            tags.append("code")
        if any(w in name for w in ["vision", "vl", "omni"]):
            tags.append("vision")
        if any(w in name for w in ["reasoning", "think"]):
            tags.append("reasoning")
        if any(w in name for w in ["mini", "small", "nano", "flash"]):
            tags.append("small")
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
            "gguf_path":     "openrouter",
            "modelfile_path": "openrouter",
            "error":         self.error,
            "or_model_id":   self.or_model_id,
            "or_name":       self.or_name,
            "or_pricing":    self.or_pricing,
        }

    def to_api_dict(self) -> dict:
        return {
            "id":       self.name,
            "object":   "model",
            "created":  int(time.time()),
            "owned_by": "openrouter",
            "permission": [],
            "root":     self.name,
            "parent":   None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# OpenRouterAdapter
# ─────────────────────────────────────────────────────────────────────────────

class OpenRouterAdapter:
    """
    Polls OpenRouter API and injects discovered models into the TEE registry.
    Runs as a background thread.
    """

    def __init__(self, registry, api_key: str, free_only: bool = True):
        self._registry  = registry
        self._api_key   = api_key
        self._free_only = free_only
        self._running   = False
        self._thread    = None
        self._known     = set()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="tee-openrouter-adapter",
            daemon=True,
        )
        self._thread.start()
        log.info(f"OpenRouter adapter started — free_only:{self._free_only}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _fetch_models(self) -> list:
        try:
            req = urllib.request.Request(
                f"{OPENROUTER_API}/models",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "HTTP-Referer":  "https://github.com/Trinity963/TEE",
                    "X-Title":       "TEE Trinity Execution Engine",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            models = data.get("data", [])
            if self._free_only:
                models = [
                    m for m in models
                    if str(m.get("pricing", {}).get("completion", "1")) == "0"
                ]
            return models
        except Exception as e:
            log.warning(f"OpenRouter unreachable: {e}")
            return []

    def _poll_loop(self):
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
            entry = OpenRouterModelEntry(m, self._api_key)
            current_names.add(entry.name)
            if entry.name not in self._known:
                self._inject(entry)
                self._known.add(entry.name)

        gone = self._known - current_names
        for name in gone:
            self._registry.remove_model(name)
            self._known.discard(name)
            log.info(f"OpenRouter model removed: {name}")

    def _inject(self, entry: OpenRouterModelEntry):
        with self._registry._lock:
            existing = self._registry._models.get(entry.name)
            if existing:
                entry.status    = existing.status
                entry.loaded_at = existing.loaded_at
            self._registry._models[entry.name] = entry
        log.info(
            f"✓ OpenRouter: {entry.name}  "
            f"[ctx:{entry.context:,}  tags:{entry.tags}]"
        )
