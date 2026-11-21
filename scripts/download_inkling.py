r"""
Resumable 12-worker parallel HTTP Range downloader & SHA-256 verifier for
Inkling-2-Mini-UD-Q4_K_XL (Mainline_Support).
Preserves any existing curl/sequential prefix in C:\models\Inkling-2-Mini-UD-Q4_K_XL.gguf.part,
tracks completed chunks in .chunks.json, verifies exact byte size + SHA-256, and atomically publishes .gguf.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/resolve/main/"
    "Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file"
)
EXPECTED_BYTES = 17_629_760_736
EXPECTED_SHA256 = "e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3"

DEST_PATH = Path(r"C:\models\Inkling-2-Mini-UD-Q4_K_XL.gguf")
PART_PATH = Path(r"C:\models\Inkling-2-Mini-UD-Q4_K_XL.gguf.part")
STATE_PATH = Path(r"C:\models\Inkling-2-Mini-UD-Q4_K_XL.gguf.chunks.json")


def chunk_bounds(idx: int, chunk_bytes: int, total_bytes: int = EXPECTED_BYTES) -> tuple[int, int, int]:
    start = idx * chunk_bytes
    end = min(total_bytes - 1, start + chunk_bytes - 1)
    length = end - start + 1
    return start, end, length


def total_bytes_for_chunks(completed: set[int] | list[int], chunk_bytes: int) -> int:
    total = 0
    for idx in completed:
        _, _, length = chunk_bounds(int(idx), chunk_bytes)
        total += length
    return total


def sha256_streaming(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


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


def load_or_init_state(chunk_bytes: int) -> dict:
    total_chunks = (EXPECTED_BYTES + chunk_bytes - 1) // chunk_bytes
    if STATE_PATH.is_file():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state.get("expected_bytes") != EXPECTED_BYTES or state.get("chunk_bytes") != chunk_bytes:
            raise RuntimeError(
                f"{STATE_PATH} chunk geometry mismatch "
                f"(got expected_bytes={state.get('expected_bytes')}, chunk_bytes={state.get('chunk_bytes')}; "
                f"need {EXPECTED_BYTES}, {chunk_bytes})."
            )
        return state

    completed: list[int] = []
    if PART_PATH.is_file():
        cur_sz = PART_PATH.stat().st_size
        if cur_sz > EXPECTED_BYTES:
            raise RuntimeError(f"Existing {PART_PATH} exceeds EXPECTED_BYTES ({cur_sz:,} > {EXPECTED_BYTES:,}).")
        complete_prefix = cur_sz // chunk_bytes
        if complete_prefix > 0:
            completed = list(range(complete_prefix))

    state = {
        "expected_bytes": EXPECTED_BYTES,
        "chunk_bytes": chunk_bytes,
        "total_chunks": total_chunks,
        "completed": completed,
    }
    save_state_atomic(state)
    return state


def save_state_atomic(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(state, f)
    os.replace(tmp, STATE_PATH)


def print_status(default_chunk_mb: int) -> int:
    if DEST_PATH.is_file():
        sz = DEST_PATH.stat().st_size
        pct = (sz / EXPECTED_BYTES) * 100.0
        print(f"[DONE] {DEST_PATH} | {sz:,} / {EXPECTED_BYTES:,} B ({pct:.2f}%)")
        return 0
    if STATE_PATH.is_file():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        cb = int(state.get("chunk_bytes", default_chunk_mb * 1024 * 1024))
        comp = state.get("completed", [])
        done_b = total_bytes_for_chunks(comp, cb)
        pct = (done_b / EXPECTED_BYTES) * 100.0
        print(
            f"[IN_PROGRESS] {PART_PATH} | {len(comp)}/{state.get('total_chunks')} chunks | "
            f"{done_b:,} / {EXPECTED_BYTES:,} B ({pct:.2f}%, {done_b / 1024**3:.2f} GiB)"
        )
        return 0
    if PART_PATH.is_file():
        sz = PART_PATH.stat().st_size
        pct = (sz / EXPECTED_BYTES) * 100.0
        print(f"[IN_PROGRESS] {PART_PATH} | {sz:,} / {EXPECTED_BYTES:,} B ({pct:.2f}%, {sz / 1024**3:.2f} GiB)")
        return 0
    print("[NOT_STARTED] Neither .gguf nor .part exists yet.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="12-worker parallel Range downloader for Inkling-2-Mini-UD-Q4_K_XL.")
    parser.add_argument("--workers", type=int, default=12, help="Parallel Range worker threads (default: 12).")
    parser.add_argument("--chunk-mb", type=int, default=32, help="Chunk size in MiB (default: 32).")
    parser.add_argument("--heartbeat-sec", type=int, default=15, help="Seconds between progress prints (default: 15).")
    parser.add_argument("--status", action="store_true", help="Print current download progress and exit.")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing final .gguf size and SHA-256.")
    args = parser.parse_args()

    if args.status:
        return print_status(args.chunk_mb)

    if args.verify_only:
        return 0 if verify_file(DEST_PATH) else 1

    DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DEST_PATH.is_file():
        if DEST_PATH.stat().st_size == EXPECTED_BYTES and verify_file(DEST_PATH):
            print(f"[OK] Already downloaded & verified: {DEST_PATH}")
            return 0
        raise RuntimeError(f"Existing {DEST_PATH} did not pass size/SHA-256 check.")

    chunk_bytes = max(1, args.chunk_mb) * 1024 * 1024
    state = load_or_init_state(chunk_bytes)
    total_chunks = int(state["total_chunks"])
    completed_set: set[int] = set(int(i) for i in state.get("completed", []))
    initial_done_bytes = total_bytes_for_chunks(completed_set, chunk_bytes)
    remaining = [idx for idx in range(total_chunks) if idx not in completed_set]

    print(
        f"Starting {args.workers}-worker parallel Range download -> {PART_PATH}\n"
        f"  Source: {URL}\n"
        f"  Target: {EXPECTED_BYTES:,} B ({EXPECTED_BYTES / 1024**3:.2f} GiB / {EXPECTED_BYTES / 1e9:.2f} GB)\n"
        f"  Chunk size: {args.chunk_mb} MiB ({total_chunks} total chunks)\n"
        f"  Preserved completed: {len(completed_set)} chunks ({initial_done_bytes / 1024**3:.2f} GiB) | "
        f"Remaining: {len(remaining)} chunks",
        flush=True,
    )

    if not PART_PATH.exists():
        with open(PART_PATH, "wb"):
            pass

    file_lock = threading.Lock()
    progress_lock = threading.Lock()
    completed_bytes = initial_done_bytes
    in_flight_map: dict[int, int] = {}
    stop_event = threading.Event()
    t_start = time.time()

    with open(PART_PATH, "r+b") as fh:
        def download_chunk(idx: int) -> None:
            nonlocal completed_bytes
            start, end, expected_len = chunk_bounds(idx, chunk_bytes)
            last_err: Exception | None = None

            for attempt in range(6):
                if stop_event.is_set():
                    return
                with progress_lock:
                    in_flight_map[idx] = 0
                written = 0
                try:
                    req = urllib.request.Request(
                        URL,
                        headers={
                            "User-Agent": "ENDO-Inkling-Downloader/2.0",
                            "Range": f"bytes={start}-{end}",
                        },
                    )
                    with urllib.request.urlopen(req, timeout=45) as resp:
                        status = getattr(resp, "status", 200)
                        if status == 200 and expected_len != EXPECTED_BYTES:
                            raise IOError(f"Server ignored Range header on chunk {idx} (HTTP 200)")
                        if status not in (200, 206):
                            raise IOError(f"Unexpected HTTP {status} on chunk {idx}")

                        while written < expected_len:
                            to_read = min(1024 * 1024, expected_len - written)
                            block = resp.read(to_read)
                            if not block:
                                break
                            offset = start + written
                            with file_lock:
                                fh.seek(offset)
                                fh.write(block)
                            written += len(block)
                            with progress_lock:
                                in_flight_map[idx] = written

                    if written != expected_len:
                        raise IOError(f"Short read on chunk {idx}: got {written} B, expected {expected_len} B")

                    with progress_lock:
                        in_flight_map.pop(idx, None)
                        completed_set.add(idx)
                        completed_bytes += expected_len
                        state["completed"] = sorted(completed_set)
                        save_state_atomic(state)
                    return
                except Exception as exc:
                    last_err = exc
                    with progress_lock:
                        in_flight_map.pop(idx, None)
                    backoff = min(30, 2 ** attempt)
                    time.sleep(backoff)

            raise RuntimeError(f"Chunk {idx} ({start}-{end}) failed after 6 attempts: {last_err}")

        def heartbeat() -> None:
            prev_t = t_start
            prev_b = initial_done_bytes
            interval = max(5, args.heartbeat_sec)
            while not stop_event.wait(interval):
                now = time.time()
                with progress_lock:
                    cur_b = completed_bytes + sum(in_flight_map.values())
                    done_chunks = len(completed_set)
                dt = max(0.001, now - prev_t)
                inst_mb_s = max(0.0, (cur_b - prev_b) / (1024 * 1024) / dt)
                elapsed = max(0.001, now - t_start)
                avg_b_s = max(1.0, (cur_b - initial_done_bytes) / elapsed)
                rem_b = max(0, EXPECTED_BYTES - cur_b)
                eta_s = int(rem_b / avg_b_s) if cur_b > initial_done_bytes else -1
                eta_str = f"{eta_s // 60}m{eta_s % 60:02d}s" if eta_s >= 0 else "n/a"
                pct = (cur_b / EXPECTED_BYTES) * 100.0
                print(
                    f"[PROGRESS] {cur_b / 1024**3:.2f}/{EXPECTED_BYTES / 1024**3:.2f} GiB ({pct:.1f}%) "
                    f"| chunks {done_chunks}/{total_chunks} | {inst_mb_s:.1f} MB/s "
                    f"(avg {avg_b_s / (1024 * 1024):.1f} MB/s) | ETA {eta_str}",
                    flush=True,
                )
                prev_t = now
                prev_b = cur_b

        hb = threading.Thread(target=heartbeat, daemon=True)
        hb.start()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = [pool.submit(download_chunk, idx) for idx in remaining]
                for fut in concurrent.futures.as_completed(futures):
                    fut.result()
        except Exception:
            stop_event.set()
            raise
        finally:
            stop_event.set()
            hb.join(timeout=2)
            fh.flush()

    if not verify_file(PART_PATH):
        return 1

    os.replace(PART_PATH, DEST_PATH)
    STATE_PATH.unlink(missing_ok=True)
    total_sec = time.time() - t_start
    avg_mb_s = ((EXPECTED_BYTES - initial_done_bytes) / (1024 * 1024)) / max(0.1, total_sec)
    print(f"[COMPLETE] {DEST_PATH} ready in {total_sec:.1f}s (avg {avg_mb_s:.1f} MB/s).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
