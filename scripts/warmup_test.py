#!/usr/bin/env python3
"""Practical warm-up / first-token latency test for a live llama-server.

The synthetic `llama-bench -p 512 -d N` sweep answers "how fast is prefill at
depth N". It does not answer the question that actually matters for agent work:
"how long until the first token arrives in a real session, and how much does the
prefix cache save me on the next turn".

This script drives a running llama-server over its HTTP API and measures, for a
ladder of context sizes, three states:

  cold      - unique prompt, nothing cached. This is the worst case: a session
              started from scratch at that depth.
  repeat    - byte-identical prompt sent again. Shows the full-cache-hit ceiling
              (should be near-instant prefill).
  new_turn  - same prefix plus one new question appended. This is the realistic
              agent turn: the conversation is already cached and only the new
              user message has to be prefilled.

All numbers come from the server's own `timings` block, so nothing is hand-timed.

Usage:
    uv run python scripts/warmup_test.py
    uv run python scripts/warmup_test.py --sizes 2048,8192 --port 8088
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"

# Technical prose reads closer to 3.6 chars/token than the usual 4.
CHARS_PER_TOKEN = 3.6

FILLER = (
    "The Windows event log pipeline records provider-level telemetry for each "
    "scheduled task, and the driver stack negotiates power states independently "
    "of the runtime scheduler, so a slow first token is usually caused by prompt "
    "prefill rather than by decode throughput on this class of hardware. "
)


def build_prompt(target_tokens: int, nonce: str) -> str:
    """Grow a distinctive prompt until it is roughly `target_tokens` long.

    The nonce goes first so each size has a unique token prefix; without it the
    server would happily reuse the previous, shorter prompt as a cached prefix.
    """
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    head = f"[run {nonce} target {target_tokens}]\n"
    reps = max(1, (target_chars - len(head)) // len(FILLER) + 1)
    return head + (FILLER * reps)


def wait_for_health(base_url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


def complete(base_url: str, prompt: str, n_predict: int, timeout_s: float) -> dict:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": True,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    wall_ms = (time.perf_counter() - started) * 1000.0
    timings = body.get("timings") or {}
    return {
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_ms": timings.get("predicted_ms"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "wall_ms": round(wall_ms, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--sizes", default="2048,8192,16384,30000")
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--request-timeout", type=float, default=1800.0)
    ap.add_argument("--health-timeout", type=float, default=900.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    print(f"waiting for {base_url}/health ...", flush=True)
    if not wait_for_health(base_url, args.health_timeout):
        print("server did not become healthy", file=sys.stderr)
        return 2
    print("server healthy\n", flush=True)

    nonce = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    for size in sizes:
        prompt = build_prompt(size, nonce)
        row = {"target_tokens": size}

        print(f"--- target {size} tokens: cold ---", flush=True)
        cold = complete(base_url, prompt, args.n_predict, args.request_timeout)
        row["cold"] = cold

        print(f"--- target {size} tokens: identical repeat (cache hit) ---", flush=True)
        row["repeat"] = complete(base_url, prompt, args.n_predict, args.request_timeout)

        followup = (
            prompt
            + "\n\nIgnore the filler above. Reply with the single word: cached"
        )
        print(f"--- target {size} tokens: new turn on cached prefix ---", flush=True)
        row["new_turn"] = complete(base_url, followup, args.n_predict, args.request_timeout)

        results.append(row)
        c = row["cold"]
        print(
            f"    cold: {c['prompt_n']} tok prefilled in "
            f"{(c['prompt_ms'] or 0)/1000:.1f}s "
            f"({c['prompt_per_second']:.1f} tok/s) -> "
            f"decode {c['predicted_per_second']:.1f} tok/s\n",
            flush=True,
        )

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = pathlib.Path(args.out) if args.out else REPORTS_DIR / f"warmup_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "kind": "warmup_first_token_latency",
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "endpoint": base_url,
        "n_predict": args.n_predict,
        "note": "all timings are server-reported /completion timings",
        "runs": results,
    }
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"{'target':>8} {'state':>9} {'prompt_n':>9} {'prefill_s':>10} {'pp_t/s':>8} {'tg_t/s':>8}")
    for row in results:
        for state in ("cold", "repeat", "new_turn"):
            t = row[state]
            pp_s = (t["prompt_ms"] or 0) / 1000.0
            print(
                f"{row['target_tokens']:>8} {state:>9} {t['prompt_n']:>9} "
                f"{pp_s:>10.2f} {(t['prompt_per_second'] or 0):>8.1f} "
                f"{(t['predicted_per_second'] or 0):>8.2f}"
            )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
