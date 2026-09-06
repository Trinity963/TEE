"""
TEE — Trinity's Execution Engine
core/gateway.py

OpenAI-compatible API gateway.
Every project that speaks the OpenAI protocol talks to TEE through here.
MiniTrini, Ethica, TBS, SARA — point them here. Nothing else changes.

Endpoints:
  GET  /v1/models
  POST /v1/chat/completions
  POST /v1/embeddings
  GET  /health
  GET  /status

MIT License — open source, sovereign, forever.
"""

import json
import logging
import time
import threading
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from registry import Registry
from runtime  import Runtime

# ── Logging ───────────────────────────────────────────────────────────────────

log = logging.getLogger("tee.gateway")

# ── Default gateway config ────────────────────────────────────────────────────

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765

# ── Backend readiness — how long to wait for a backend to become ready ────────

BACKEND_READY_TIMEOUT  = 30   # seconds
BACKEND_READY_INTERVAL = 0.5  # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _json_response(handler, status: int, data: dict):
    """Write a JSON response."""
    body = json.dumps(data, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type",   "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.end_headers()
    handler.wfile.write(body)


def _error(handler, status: int, message: str, code: str = "tee_error"):
    """Write an OpenAI-compatible error response."""
    _json_response(handler, status, {
        "error": {
            "message": message,
            "type":    code,
            "code":    status,
        }
    })


def _read_body(handler) -> Optional[dict]:
    """Read and parse the JSON request body."""
    try:
        length = int(handler.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = handler.rfile.read(length)
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        log.error(f"Failed to read request body: {e}")
        return None


def _proxy_request_openrouter(entry, body: dict) -> tuple:
    """Forward a chat completion request to OpenRouter API."""
    try:
        payload = dict(body)
        payload["model"] = entry.or_model_id
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            f"{entry.or_base}/chat/completions",
            data=data,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {entry.or_api_key}",
                "HTTP-Referer":  "https://github.com/Trinity963/TEE",
                "X-Title":       "TEE Trinity Execution Engine",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw      = resp.read()
            response = json.loads(raw.decode("utf-8"))
            return response, resp.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw), e.code
        except Exception:
            return {"error": {"message": raw}}, e.code
    except Exception as e:
        return {"error": {"message": str(e)}}, 502


def _wait_for_backend(url: str, timeout: float = BACKEND_READY_TIMEOUT) -> bool:
    """Poll a backend health endpoint until it responds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.urlopen(f"{url}/health", timeout=2)
            if req.status == 200:
                return True
        except Exception:
            pass
        time.sleep(BACKEND_READY_INTERVAL)
    return False


def _proxy_request(url: str, body: dict) -> tuple:
    """
    Forward a request to a backend server.
    Returns (response_dict, status_code).
    """
    try:
        data    = json.dumps(body).encode("utf-8")
        req     = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw      = resp.read()
            response = json.loads(raw.decode("utf-8"))
            return response, resp.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw), e.code
        except Exception:
            return {"error": {"message": raw}}, e.code
    except Exception as e:
        return {"error": {"message": str(e)}}, 502


# ─────────────────────────────────────────────────────────────────────────────
# Request Handler
# ─────────────────────────────────────────────────────────────────────────────

class TEEHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for the TEE gateway.
    Registry and runtime are injected via the server instance.
    """

    # Suppress default request logging — TEE logs its own way
    def log_message(self, fmt, *args):
        pass

    # ── CORS preflight ────────────────────────────────────────────────────────

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    # ── GET ───────────────────────────────────────────────────────────────────

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/v1/models":
            self._handle_list_models()
        elif path == "/v1/models/downloads":
            self._handle_downloads_list()
        elif path == "/health":
            self._handle_health()
        elif path == "/status":
            self._handle_status()
        else:
            _error(self, 404, f"Unknown endpoint: {path}", "not_found")

    # ── POST ──────────────────────────────────────────────────────────────────

    def do_POST(self):
        path = self.path.split("?")[0]

        if path == "/v1/chat/completions":
            self._handle_chat()
        elif path == "/v1/embeddings":
            self._handle_embeddings()
        elif path == "/v1/models/load":
            self._handle_load()
        elif path == "/v1/models/unload":
            self._handle_unload()
        elif path == "/v1/models/download":
            self._handle_download()
        else:
            _error(self, 404, f"Unknown endpoint: {path}", "not_found")

    def _handle_downloads_list(self):
        """GET /v1/models/downloads — list all downloads and their progress."""
        if not self.server.downloader:
            _error(self, 503, "Downloader not available.", "downloader_unavailable")
            return
        _json_response(self, 200, {"downloads": self.server.downloader.list_downloads()})

    def _handle_download(self):
        """POST /v1/models/download — start a HuggingFace download."""
        if not self.server.downloader:
            _error(self, 503, "Downloader not available.", "downloader_unavailable")
            return
        body = _read_body(self)
        if not body:
            _error(self, 400, "Body required: {repo_id, filename}", "invalid_request")
            return
        repo_id  = body.get("repo_id", "")
        filename = body.get("filename", "")
        if not repo_id or not filename:
            _error(self, 400, "Fields 'repo_id' and 'filename' are required.", "invalid_request")
            return
        log.info(f"POST /v1/models/download — {repo_id}/{filename}")
        download_id = self.server.downloader.start_download(repo_id, filename)
        _json_response(self, 200, {
            "status":      "started",
            "download_id": download_id,
            "repo_id":     repo_id,
            "filename":    filename,
        })

    def _handle_load(self):
        """POST /v1/models/load — load a model into a backend."""
        body = _read_body(self)
        if not body:
            _error(self, 400, "Field 'model' is required", "invalid_request")
            return
        model_name = body.get("model", "")
        if not model_name:
            _error(self, 400, "Field 'model' is required", "invalid_request")
            return
        log.info(f"POST /v1/models/load — model:{model_name}")
        loaded = self.server.runtime.load(model_name)
        if loaded is None:
            entry = self.server.registry.get_model(model_name)
            if entry is None:
                _error(self, 404, f"Model '{model_name}' not found.", "model_not_found")
            else:
                _error(self, 503, f"Model '{model_name}' could not be loaded.", "backend_unavailable")
            return
        _json_response(self, 200, {"status": "loaded", "model": loaded.to_dict()})

    def _handle_unload(self):
        """POST /v1/models/unload — unload a model from its backend."""
        body = _read_body(self)
        if not body:
            _error(self, 400, "Field 'model' is required", "invalid_request")
            return
        model_name = body.get("model", "")
        if not model_name:
            _error(self, 400, "Field 'model' is required", "invalid_request")
            return
        log.info(f"POST /v1/models/unload — model:{model_name}")
        loaded = self.server.runtime.get_loaded(model_name)
        if loaded is None:
            _error(self, 404, f"Model '{model_name}' is not loaded.", "model_not_loaded")
            return
        self.server.runtime._unload(model_name)
        _json_response(self, 200, {"status": "unloaded", "model": model_name})

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_list_models(self):
        """GET /v1/models — OpenAI-compatible model list."""
        result = self.server.registry.list_models_api()
        log.info(f"GET /v1/models — {result['data'].__len__()} models")
        _json_response(self, 200, result)

    def _handle_health(self):
        """GET /health — simple liveness check."""
        _json_response(self, 200, {
            "status":  "ok",
            "service": "TEE — Trinity's Execution Engine",
            "version": "1.0.0",
        })

    def _handle_status(self):
        """GET /status — full system status."""
        status = self.server.runtime.system_status()
        status["registry"] = self.server.registry.get_manifest()
        log.info("GET /status")
        _json_response(self, 200, status)

    def _handle_chat(self):
        """POST /v1/chat/completions — chat inference."""
        body = _read_body(self)
        if body is None:
            _error(self, 400, "Invalid JSON body", "invalid_request")
            return

        model_name = body.get("model", "")
        if not model_name:
            _error(self, 400, "Field 'model' is required", "invalid_request")
            return

        log.info(f"POST /v1/chat/completions — model:{model_name}")

        # Resolve model — get or load
        loaded = self.server.runtime.get_or_load(model_name)
        if loaded is None:
            # Check if model exists in registry at all
            entry = self.server.registry.get_model(model_name)
            if entry is None:
                _error(
                    self, 404,
                    f"Model '{model_name}' not found. "
                    f"Available: {[m.name for m in self.server.registry.list_models()]}",
                    "model_not_found",
                )
            else:
                _error(
                    self, 503,
                    f"Model '{model_name}' could not be loaded. "
                    f"Check that a backend (llama.cpp / vllm) is installed.",
                    "backend_unavailable",
                )
            return

        # Ollama and OpenRouter are always ready — skip health wait
        if loaded.backend not in ("ollama", "openrouter"):
            ready = _wait_for_backend(loaded.base_url())
            if not ready:
                _error(
                    self, 503,
                    f"Backend for '{model_name}' did not become ready in time.",
                    "backend_timeout",
                )
                return

        # OpenRouter — proxy directly to openrouter.ai
        if loaded.backend == "openrouter":
            or_entry = getattr(loaded, "_or_entry", None)
            if or_entry is None:
                or_entry = self.server.registry.get_model(model_name)
            if or_entry is None:
                _error(self, 503, f"OpenRouter entry missing for '{model_name}'.", "backend_unavailable")
                return
            response, status = _proxy_request_openrouter(or_entry, body)
            loaded.touch()
            _json_response(self, status, response)
            return

        # Apply model defaults if not overridden in request
        entry = self.server.registry.get_model(model_name)
        if entry:
            defaults = entry.data.get("defaults", {})
            for key, val in defaults.items():
                if key not in body:
                    body[key] = val

        # Forward to backend
        backend_url = f"{loaded.base_url()}/v1/chat/completions"
        response, status = _proxy_request(backend_url, body)

        loaded.touch()
        _json_response(self, status, response)

    def _handle_embeddings(self):
        """POST /v1/embeddings — embeddings inference."""
        body = _read_body(self)
        if body is None:
            _error(self, 400, "Invalid JSON body", "invalid_request")
            return

        model_name = body.get("model", "")
        if not model_name:
            _error(self, 400, "Field 'model' is required", "invalid_request")
            return

        log.info(f"POST /v1/embeddings — model:{model_name}")

        loaded = self.server.runtime.get_or_load(model_name)
        if loaded is None:
            _error(self, 503, f"Model '{model_name}' could not be loaded.", "backend_unavailable")
            return

        if loaded.backend != "ollama":
            ready = _wait_for_backend(loaded.base_url())
            if not ready:
                _error(self, 503, f"Backend timeout for '{model_name}'.", "backend_timeout")
                return

        backend_url = f"{loaded.base_url()}/v1/embeddings"
        response, status = _proxy_request(backend_url, body)

        loaded.touch()
        _json_response(self, status, response)


# ─────────────────────────────────────────────────────────────────────────────
# TEEServer — wires registry and runtime into the HTTP server
# ─────────────────────────────────────────────────────────────────────────────

class TEEServer(HTTPServer):
    """
    HTTPServer subclass that carries registry and runtime
    so handlers can access them without globals.
    SO_REUSEADDR enabled — port releases immediately on shutdown.
    """

    allow_reuse_address = True

    def __init__(self, host: str, port: int, registry: Registry, runtime: Runtime, downloader=None):
        super().__init__((host, port), TEEHandler)
        self.registry   = registry
        self.runtime    = runtime
        self.downloader = downloader


# ─────────────────────────────────────────────────────────────────────────────
# Gateway — public interface
# ─────────────────────────────────────────────────────────────────────────────

class Gateway:
    """
    TEE API Gateway.
    Wires registry + runtime into an OpenAI-compatible HTTP server.
    """

    def __init__(
        self,
        registry:   Registry,
        runtime:    Runtime,
        downloader = None,
        host:       str = DEFAULT_HOST,
        port:       int = DEFAULT_PORT,
    ):
        self._registry   = registry
        self._runtime    = runtime
        self._downloader = downloader
        self._host       = host
        self._port       = port
        self._server:  Optional[TEEServer]      = None
        self._thread:  Optional[threading.Thread] = None

    def start(self):
        """Start the gateway in a background thread."""
        self._server = TEEServer(
            self._host, self._port,
            self._registry, self._runtime, self._downloader,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="tee-gateway",
            daemon=True,
        )
        self._thread.start()

        log.info(f"Gateway listening on http://{self._host}:{self._port}/v1/")
        log.info(f"  GET  http://{self._host}:{self._port}/v1/models")
        log.info(f"  POST http://{self._host}:{self._port}/v1/chat/completions")
        log.info(f"  POST http://{self._host}:{self._port}/v1/embeddings")
        log.info(f"  GET  http://{self._host}:{self._port}/health")
        log.info(f"  GET  http://{self._host}:{self._port}/status")

    def stop(self):
        """Shut down the gateway."""
        if self._server:
            self._server.shutdown()
        log.info("Gateway stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI — python3 gateway.py [models_dir] [models_dir ...]
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  TEE  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dirs = sys.argv[1:] if len(sys.argv) > 1 else []

    # ── Build the stack ───────────────────────────────────────────────────────

    registry = Registry()
    for d in dirs:
        registry.add_watch_dir(d)

    print("\n" + "─" * 60)
    print("  TEE — Trinity's Execution Engine")
    print("  Sovereign LLM Router  |  MIT License")
    print("─" * 60)

    # Scan models
    print("\nScanning model directories...")
    registry.scan_all()
    registry.start_watching()

    # Start runtime
    runtime = Runtime(registry)
    runtime.start()

    # Start gateway
    gateway = Gateway(registry, runtime)
    gateway.start()

    # Print manifest
    registry.print_manifest()
    runtime.print_status()

    print("\nTEE is running. Ctrl+C to stop.\n")

    # ── Keep alive ────────────────────────────────────────────────────────────

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down TEE...")
        gateway.stop()
        runtime.stop()
        registry.stop_watching()
        print("TEE stopped. Sovereign to the end.\n")
