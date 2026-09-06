#!/usr/bin/env python3
"""
TEE — Trinity's Execution Engine
tee.py

Sovereign entry point.
Starts the full stack — registry, runtime, gateway.
One command. Everything runs.

Usage:
  python3 tee.py                          # uses tee.config
  python3 tee.py --models /path/to/models # override models dir
  python3 tee.py --port 8765              # override port
  python3 tee.py --help                   # show help

MIT License — open source, sovereign, forever.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

# ── Ensure core/ is on the path ───────────────────────────────────────────────

ROOT = Path(__file__).parent
CORE = ROOT / "core"
sys.path.insert(0, str(CORE))

from registry import Registry
from runtime    import Runtime
from gateway    import Gateway
from downloader import Downloader
sys.path.insert(0, str(Path(__file__).parent / 'adapters'))
from ollama      import OllamaAdapter
from openrouter  import OpenRouterAdapter
sys.path.insert(0, str(Path(__file__).parent / 'ui'))
import server as ui_server

# ── Version ───────────────────────────────────────────────────────────────────

TEE_VERSION = "1.0.0"

# ── Default config path ───────────────────────────────────────────────────────

DEFAULT_CONFIG = ROOT / "tee.config"
CONFIG_EXAMPLE = ROOT / "tee.config.example"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  TEE  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("tee")


# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────

BANNER = f"""
╔══════════════════════════════════════════════════════════════╗
║         TEE — Trinity's Execution Engine  v{TEE_VERSION}           ║
║         Sovereign LLM Router  |  MIT License                 ║
║         github.com/YOU/TEE                                   ║
╚══════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    """
    Load tee.config.
    If not found — offer to create one from the example.
    """
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            log.info(f"Config loaded: {config_path}")
            return config
        except Exception as e:
            log.error(f"Failed to read config: {e}")
            sys.exit(1)

    # No config found
    log.warning(f"Config not found: {config_path}")

    if CONFIG_EXAMPLE.exists():
        print(f"\n  No tee.config found.")
        print(f"  Creating one from tee.config.example...")
        import shutil
        shutil.copy(CONFIG_EXAMPLE, config_path)
        print(f"  Created: {config_path}")
        print(f"  Edit it to set your models directories, then restart TEE.\n")

        # Return the example config so TEE starts with defaults
        with open(config_path) as f:
            return json.load(f)
    else:
        log.warning("No config and no example found — starting with defaults")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# First-run setup
# ─────────────────────────────────────────────────────────────────────────────

def first_run_setup(config: dict) -> dict:
    """
    If no models_dirs configured — ask the user where their models live.
    Plain English. No jargon.
    """
    dirs = config.get("models_dirs", [])
    valid_dirs = [d for d in dirs if d.get("path") and d["path"] != "/path/to/your/models"]

    if valid_dirs:
        return config  # Already configured

    print("\n" + "─" * 60)
    print("  Welcome to TEE — first run setup")
    print("─" * 60)
    print("\n  Where are your models stored?")
    print("  (This is the folder where your .gguf files live)\n")

    dirs_config = []

    while True:
        path_input = input("  Model directory path: ").strip()

        if not path_input:
            print("  Path cannot be empty.")
            continue

        p = Path(path_input).expanduser().resolve()

        if not p.exists():
            print(f"  Directory not found: {p}")
            create = input("  Create it? [y/N]: ").strip().lower()
            if create == "y":
                p.mkdir(parents=True, exist_ok=True)
                print(f"  Created: {p}")
            else:
                continue

        dirs_config.append({
            "path":          str(p),
            "label":         "auto",
            "detected_type": "auto",
            "user_renamed":  False,
            "watch":         True,
        })

        print(f"  ✓ Added: {p}")

        another = input("\n  Add another directory? [y/N]: ").strip().lower()
        if another != "y":
            break

    config["models_dirs"] = dirs_config

    # Save updated config
    try:
        with open(DEFAULT_CONFIG, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\n  Config saved → {DEFAULT_CONFIG}")
    except Exception as e:
        log.warning(f"Could not save config: {e}")

    print()
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        prog="tee",
        description="TEE — Trinity's Execution Engine. Sovereign LLM router.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tee.py
  python3 tee.py --models /mnt/nvme/models
  python3 tee.py --models /mnt/nvme/models --models /mnt/hdd/models
  python3 tee.py --port 8765 --host 0.0.0.0
  python3 tee.py --config /path/to/tee.config
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to tee.config (default: ./tee.config)",
    )
    parser.add_argument(
        "--models", "-m",
        type=str,
        action="append",
        metavar="DIR",
        help="Model directory to watch (can be used multiple times)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Gateway port (default: 8765)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Gateway host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version and exit",
    )
    parser.add_argument(
        "--no-watch",
        action="store_true",
        help="Disable directory watching (scan once only)",
    )
    parser.add_argument(
        "--idle-unload",
        type=int,
        default=None,
        metavar="MINUTES",
        help="Unload idle models after N minutes (default: 30)",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Version flag
    if args.version:
        print(f"TEE v{TEE_VERSION}")
        sys.exit(0)

    # Banner
    print(BANNER)

    # Load config
    config_path = Path(args.config)
    config      = load_config(config_path)

    # First run setup if needed and no --models flag
    if not args.models:
        config = first_run_setup(config)

    # CLI overrides
    if args.port:
        config.setdefault("gateway", {})["port"] = args.port
    if args.host:
        config.setdefault("gateway", {})["host"] = args.host
    if args.idle_unload:
        config.setdefault("runtime", {})["unload_idle_after_minutes"] = args.idle_unload

    # Resolve gateway settings
    gw_host = config.get("gateway", {}).get("host", "0.0.0.0")
    gw_port = config.get("gateway", {}).get("port", 8765)

    # ── Build the stack ───────────────────────────────────────────────────────

    registry = Registry(str(config_path) if not args.models else None)

    # CLI --models dirs take priority over config
    if args.models:
        for d in args.models:
            registry.add_watch_dir(d, label="auto", watch=not args.no_watch)
    else:
        # Ensure watch flag respected
        if args.no_watch:
            for entry in registry._watched:
                entry["watch"] = False

    # Scan all directories
    log.info("Scanning model directories...")
    registry.scan_all()

    if not args.no_watch:
        registry.start_watching()

    # Start Ollama adapter
    ollama_adapter = OllamaAdapter(registry)
    ollama_adapter.start()

    # Start OpenRouter adapter if configured
    or_config   = config.get("openrouter", {})
    or_api_key  = or_config.get("api_key", "")
    or_enabled  = or_config.get("enabled", False)
    or_free_only = or_config.get("free_only", True)
    openrouter_adapter = None
    if or_enabled and or_api_key and or_api_key != "YOUR_KEY_HERE":
        openrouter_adapter = OpenRouterAdapter(registry, or_api_key, or_free_only)
        openrouter_adapter.start()
    else:
        log.info("OpenRouter adapter disabled — set openrouter.enabled and api_key in tee.config")
    # Start runtime
    runtime = Runtime(registry, config)
    runtime.start()

    # Start downloader
    gguf_dir = next(
        (d["path"] for d in config.get("models_dirs", []) if "gguf" in d.get("path","").lower()),
        "/srv/LLMs/gguf"
    )
    downloader = Downloader(gguf_dir)

    # Start gateway
    gateway = Gateway(registry, runtime, downloader, host=gw_host, port=gw_port)
    gateway.start()

    # Start UI server
    ui_host = "0.0.0.0"
    ui_port = 8766
    import threading
    ui_thread = threading.Thread(
        target=ui_server.run,
        kwargs={"host": ui_host, "port": ui_port},
        name="tee-ui",
        daemon=True,
    )
    ui_thread.start()
    # Print state
    registry.print_manifest()
    runtime.print_status()

    # ── Ready ─────────────────────────────────────────────────────────────────

    print("─" * 60)
    print(f"  TEE is running — http://{gw_host}:{gw_port}/v1/")
    print(f"  TEE UI running  — http://{ui_host}:{ui_port}/")
    print(f"  Models registered: {registry.model_count()}")
    print(f"  Drop a .gguf into any watched directory to add a model.")
    print(f"  Ctrl+C to stop.")
    print("─" * 60 + "\n")

    # ── Signal handling ───────────────────────────────────────────────────────

    def shutdown(sig, frame):
        print("\n\nShutting down TEE...")
        gateway.stop()
        runtime.stop()
        ollama_adapter.stop()
        if openrouter_adapter:
            openrouter_adapter.stop()
        registry.stop_watching()
        print("TEE stopped. Sovereign to the end.\n")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Keep alive ────────────────────────────────────────────────────────────

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
