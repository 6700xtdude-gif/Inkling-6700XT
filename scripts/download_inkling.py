r"""
Resumable downloader & SHA-256 verifier for Inkling-2-Mini-UD-Q4_K_XL (Mainline_Support).
Downloads into C:\models\Inkling-2-Mini-UD-Q4_K_XL.gguf.part via curl.exe (-C -),
verifies exact byte size and SHA-256, and atomically renames to .gguf.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

URL = (
    "https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/"
    "Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file"
)
EXPECTED_BYTES = 17_629_760_736
EXPECTED_SHA256 = "e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3"

DEST_PATH = Path(r"C:\models\Inkling-2-Mini-UD-Q4_K_XL.gguf")
PART_PATH = Path(r"C:\models\Inkling-2-Mini-UD-Q4_K_XL.gguf.part")


def sha256_streaming(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def print_status() -> int:
    if DEST_PATH.is_file():
        sz = DEST_PATH.stat().st_size
        pct = (sz / EXPECTED_BYTES) * 100.0
        print(f"[DONE] {DEST_PATH} | {sz:,} / {EXPECTED_BYTES:,} B ({pct:.2f}%)")
        return 0
    if PART_PATH.is_file():
        sz = PART_PATH.stat().st_size
        pct = (sz / EXPECTED_BYTES) * 100.0
        print(f"[IN_PROGRESS] {PART_PATH} | {sz:,} / {EXPECTED_BYTES:,} B ({pct:.2f}%, {sz / 1024**3:.2f} GiB)")
        return 0
    print("[NOT_STARTED] Neither .gguf nor .part exists yet.")
    return 0


def verify_file(path: Path) -> bool:
    if not path.is_file():
        print(f"[ERROR] File missing: {path}", file=sys.stderr)
        return False
    sz = path.stat().st_size
    if sz != EXPECTED_BYTES:
        print(f"[ERROR] Size mismatch for {path}: got {sz:,} B, expected {EXPECTED_BYTES:,} B", file=sys.stderr)
        return False
    print(f"Verifying SHA-256 of {path} ({sz / 1024**3:.2f} GiB)...", flush=True)
    actual = sha256_streaming(path)
    if actual != EXPECTED_SHA256:
        print(f"[ERROR] SHA-256 mismatch: got {actual}, expected {EXPECTED_SHA256}", file=sys.stderr)
        return False
    print(f"[OK] SHA-256 verified: {actual}", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify Inkling-2-Mini-UD-Q4_K_XL (Mainline_Support).")
    parser.add_argument("--status", action="store_true", help="Print current download progress and exit.")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing final .gguf size and SHA-256.")
    parser.add_argument("--heartbeat-sec", type=int, default=20, help="Seconds between progress prints (default: 20).")
    args = parser.parse_args()

    if args.status:
        return print_status()

    if args.verify_only:
        return 0 if verify_file(DEST_PATH) else 1

    DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DEST_PATH.is_file():
        if DEST_PATH.stat().st_size == EXPECTED_BYTES:
            if verify_file(DEST_PATH):
                print(f"[OK] Already downloaded & verified: {DEST_PATH}")
                return 0
            raise RuntimeError(f"Existing {DEST_PATH} failed SHA-256 verification.")
        raise RuntimeError(
            f"Existing {DEST_PATH} has unexpected size {DEST_PATH.stat().st_size:,} != {EXPECTED_BYTES:,}."
        )

    curl_exe = shutil.which("curl.exe") or "curl.exe"
    cmd = [
        curl_exe,
        "-L",
        "-C", "-",
        "--fail",
        "--retry", "5",
        "--retry-delay", "5",
        "--silent",
        "--show-error",
        "-o", str(PART_PATH),
        URL,
    ]

    start_sz = PART_PATH.stat().st_size if PART_PATH.is_file() else 0
    print(
        f"Starting resumable download -> {PART_PATH}\n"
        f"  Source: {URL}\n"
        f"  Target: {EXPECTED_BYTES:,} B ({EXPECTED_BYTES / 1024**3:.2f} GiB / {EXPECTED_BYTES / 1e9:.2f} GB)\n"
        f"  Resume offset: {start_sz:,} B",
        flush=True,
    )

    stop_event = threading.Event()
    t_start = time.time()

    def heartbeat() -> None:
        prev_t = t_start
        prev_sz = start_sz
        interval = max(5, args.heartbeat_sec)
        while not stop_event.wait(interval):
            now = time.time()
            cur_sz = PART_PATH.stat().st_size if PART_PATH.is_file() else 0
            dt = max(0.001, now - prev_t)
            mb_s = ((cur_sz - prev_sz) / (1024 * 1024)) / dt
            pct = (cur_sz / EXPECTED_BYTES) * 100.0
            elapsed = now - t_start
            print(
                f"[PROGRESS] {cur_sz / 1024**3:.2f}/{EXPECTED_BYTES / 1024**3:.2f} GiB ({pct:.1f}%) "
                f"| {mb_s:.1f} MB/s | elapsed {elapsed:.0f}s",
                flush=True,
            )
            prev_t = now
            prev_sz = cur_sz

    hb_thread = threading.Thread(target=heartbeat, daemon=True)
    hb_thread.start()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    finally:
        stop_event.set()
        hb_thread.join(timeout=2)

    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        print(f"[ERROR] curl.exe failed: {err}", file=sys.stderr)
        return 1

    if not verify_file(PART_PATH):
        return 1

    os.replace(PART_PATH, DEST_PATH)
    total_sec = time.time() - t_start
    avg_mb_s = ((EXPECTED_BYTES - start_sz) / (1024 * 1024)) / max(0.1, total_sec)
    print(f"[COMPLETE] {DEST_PATH} ready in {total_sec:.1f}s (avg {avg_mb_s:.1f} MB/s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
