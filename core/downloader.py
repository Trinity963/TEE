"""
TEE — Trinity's Execution Engine
core/downloader.py

HuggingFace model downloader.
Downloads GGUF files from HF into the watched directory.
Tracks progress per download. Thread-safe.
MIT License — open source, sovereign, forever.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger("tee.downloader")

# ── Download state ────────────────────────────────────────────────────────────

class DownloadState:
    def __init__(self, repo_id: str, filename: str, dest: Path):
        self.repo_id   = repo_id
        self.filename  = filename
        self.dest      = dest
        self.status    = "pending"   # pending | downloading | done | error
        self.progress  = 0.0         # 0.0 – 100.0
        self.bytes_done = 0
        self.bytes_total = 0
        self.error     = None
        self.started_at = time.time()
        self.finished_at = None

    def to_dict(self) -> dict:
        return {
            "repo_id":     self.repo_id,
            "filename":    self.filename,
            "dest":        str(self.dest),
            "status":      self.status,
            "progress":    round(self.progress, 1),
            "bytes_done":  self.bytes_done,
            "bytes_total": self.bytes_total,
            "error":       self.error,
            "started_at":  self.started_at,
            "finished_at": self.finished_at,
        }


# ── Downloader ────────────────────────────────────────────────────────────────

class Downloader:
    """
    Manages HuggingFace model downloads.
    Each download runs in its own thread.
    Progress is tracked and queryable.
    """

    def __init__(self, dest_dir: str):
        self._dest_dir = Path(dest_dir)
        self._downloads: Dict[str, DownloadState] = {}
        self._lock = threading.Lock()

    def start_download(self, repo_id: str, filename: str) -> str:
        """
        Start downloading a file from HuggingFace.
        Returns a download_id to track progress.
        """
        download_id = f"{repo_id}/{filename}".replace("/", "_").replace(" ", "_")
        dest = self._dest_dir / filename

        with self._lock:
            if download_id in self._downloads:
                state = self._downloads[download_id]
                if state.status in ("pending", "downloading"):
                    log.info(f"Download already in progress: {download_id}")
                    return download_id

            state = DownloadState(repo_id, filename, dest)
            self._downloads[download_id] = state

        thread = threading.Thread(
            target=self._run,
            args=(download_id, repo_id, filename, dest),
            name=f"tee-dl-{download_id[:20]}",
            daemon=True,
        )
        thread.start()
        log.info(f"Download started: {repo_id}/{filename} → {dest}")
        return download_id

    def _run(self, download_id: str, repo_id: str, filename: str, dest: Path):
        """Worker thread — downloads the file with progress tracking."""
        state = self._downloads[download_id]
        state.status = "downloading"

        try:
            from huggingface_hub import hf_hub_url
            import urllib.request

            url = hf_hub_url(repo_id=repo_id, filename=filename)
            log.info(f"Fetching: {url}")

            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")

            req = urllib.request.Request(url, headers={"User-Agent": "TEE/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                state.bytes_total = total
                done = 0
                chunk = 1024 * 1024  # 1MB chunks
                with open(tmp, "wb") as f:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        state.bytes_done = done
                        state.progress = (done / total * 100) if total else 0.0

            tmp.rename(dest)
            state.status = "done"
            state.progress = 100.0
            state.finished_at = time.time()
            log.info(f"✓ Download complete: {dest.name}")

        except Exception as e:
            state.status = "error"
            state.error  = str(e)
            log.error(f"Download failed: {repo_id}/{filename}: {e}")
            if dest.with_suffix(".tmp").exists():
                dest.with_suffix(".tmp").unlink()

    def get_status(self, download_id: str) -> Optional[dict]:
        with self._lock:
            state = self._downloads.get(download_id)
            return state.to_dict() if state else None

    def list_downloads(self) -> list:
        with self._lock:
            return [s.to_dict() for s in self._downloads.values()]
