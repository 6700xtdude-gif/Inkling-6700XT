# Inkling 2 Mini on AMD Radeon RX 6700 XT (12 GB VRAM + 64 GB DDR4)

**Verdict: Thumbs-up baseline — functional with `pi` coding agent, acceptably slow at ~14–15 tok/s, zero truncations at 32K.**

This repository documents the exact Windows 11 Vulkan setup, loop-aware `-ot` tensor offload strategy, `llama.cpp` router preset, `pi` client config, benchmarks, and live multi-turn agent smoke tests for running Thinking Machines Lab's **Inkling 2 Mini** (`unsloth/Inkling-2-Mini-GGUF`, `Mainline_Support` `UD-Q4_K_XL` and `UD-Q3_K_XL`) on a **12 GB AMD Radeon RX 6700 XT**.

> **Scope of every number on this page:** text-only. Inkling 2 Mini also accepts images upstream, but no `mmproj` vision projector is loaded anywhere here - see [Section 2](#modality-scope---text-only-runs-on-a-vision-capable-model). Under clean-GPU conditions the measured generation rate is **`15.4-16.2 tok/s`** (Section 4), which supersedes the `~14-15 tok/s` figure in the verdict line above.

---

## 1. Test Hardware & Software Stack

| Component | Observed Configuration |
| --- | --- |
| **GPU** | AMD Radeon RX 6700 XT 12 GB (`12272 MiB` Vulkan / `12288 MB` physical GDDR6, `gfx1031`, ReBAR enabled, AMD proprietary Vulkan driver) |
| **CPU** | AMD Ryzen 5 5600 (6C / 12T) |
| **System RAM** | 64 GB DDR4 (`65436 MiB` reported by `llama.cpp`) |
| **OS Constraint** | Windows 11 Pro — keep dedicated VRAM under ~`10.8 GB` (`11,059 MB`) to leave Windows DWM headroom and avoid WDDM GPU spillover |
| **Inference Engine** | Official mainline `llama.cpp` **`b12615`** (`a818a46f3`, `win-vulkan-x64`, includes PR `#32694` `inkling2_loop` and PR `#32704` per-expert NVFP4 scale fix) |
| **Agent Client** | `pi` (`--thinking low`, verified from a separate Windows PC over 1 GbE Ethernet against the router endpoint) |

---

## 2. Avoid the Download Trap (`Mainline_Support` vs Root GGUF)

`unsloth/Inkling-2-Mini-GGUF` hosts two sets of `UD-Q4_K_XL` / `UD-Q3_K_XL` weights with identical sizes:
- **Root `.gguf` files** target Unsloth's custom `llama.cpp` fork.
- **`Mainline_Support/*.gguf_file`** targets official upstream `ggml-org/llama.cpp` (`b12600+` with PR `#32704` merged). Download from `Mainline_Support/` and save locally as `.gguf`.

Verified local artifacts tested in this baseline:

| Quant | Variant | File Size | Verified SHA-256 |
| --- | --- | --- | --- |
| **`UD-Q4_K_XL`** *(primary)* | `Mainline_Support` | `17,629,760,736 B` (`16.42 GiB`) | `e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3` |
| **`UD-Q3_K_XL`** *(32K / fast option)* | `Mainline_Support` | `13,631,779,296 B` (`12.70 GiB`) | `e2d03e8c92ac21cff0e3309b95f40dfd27d1825184f0d2bc09db249797273639` |

*(No `mmproj` vision projector was loaded so the full GDDR6 headroom goes to looped experts, KV cache, and compute buffers.)*

### Modality Scope - Text-Only Runs on a Vision-Capable Model

**Inkling 2 Mini is natively image+text. Every number in this repository is text-only.**

| Aspect | Status |
| --- | --- |
| Upstream capability (`documented`) | `unsloth/Inkling-2-Mini-GGUF` is tagged `pipeline_tag: image-text-to-text`; the published chat template accepts `image` / `input_image` / `image_url` parts and raises on audio (*"Inkling 2 does not accept audio input"*) |
| Vision weights | Shipped as separate projectors: `mmproj-F32.gguf` (`1,273,048,512 B`), `mmproj-BF16.gguf` (`641,805,760 B`), `mmproj-F16.gguf` (`639,348,160 B`). **None downloaded, none loaded.** |
| Text GGUF | `general.architecture = inkling2_loop`, `744` tensors, no vision encoder inside - the projector is the only path to image input |
| Benchmarks / smoke tests | `observed` text-only: every `llama-bench` / `llama-server` invocation in this repo omits `--mmproj`, and [launch_inkling.ps1](scripts/launch_inkling.ps1) logs `mmproj: skipped` |

What this means for reading the results:

- **No claim is made here about image understanding, OCR, or screenshot reading.** Vision is `hypothesis` / untested on this hardware.
- **Vision does not fit the current 32K configuration.** The smallest projector (`F16`) adds `610 MiB` of VRAM plus image-token prefill cost. At 32K, `UD-Q4_K_XL` already peaks at `11,425 MB` of `12,272 MB` (Section 4), so enabling vision would require deeper CPU expert offload or a smaller context window.
- If vision ever gets tested, it must be a separate benchmark row with the projector noted - never blended into the text-only tables above.

---

## 3. Loop-Pinned Hybrid Split (`-ot` vs `--n-cpu-moe`) & `-lm none` Fix

Inkling 2 Mini has **24 physical layers (`blk.0`–`blk.23`)**, where the **middle 12 layers (`blk.6`–`blk.17`) execute 3× per token** (`6 + 3×12 + 6 = 48` logical layers).

Using naive `--n-cpu-moe 12` evicts `blk.0..11` to DDR4 — putting half of the 3× looped layers on slow system RAM while leaving 1× outer layers in GDDR6. Instead, we pin **all attention, shared/dense layers, and the 3× looped experts (`6..17`) in VRAM**, and offload only the **1× outer MoE experts (`0..5` and `18..23`)** to 64 GB DDR4:

```text
-ngl 99 -ot "blk\.([0-5]|1[8-9]|2[0-3])\.ffn_.*_exps\.=CPU"
```

### Verified Tensor Split ([inspect_inkling_tensors.py](scripts/inspect_inkling_tensors.py))

| Quant | Total Tensor Payload | Pinned in GPU VRAM | Offloaded to DDR4 (`-ot ...=CPU`) | Layout Report |
| --- | --- | --- | --- | --- |
| **`UD-Q4_K_XL`** | `16.41 GiB` (`744` tensors) | **`8.10 GiB`** (`704` tensors) | `8.31 GiB` (`40` tensors) | [tensor_layout.json](reports/tensor_layout.json) |
| **`UD-Q3_K_XL`** | `12.68 GiB` (`744` tensors) | **`7.62 GiB`** (`704` tensors) | `5.07 GiB` (`40` tensors) | [tensor_layout_q3.json](reports/tensor_layout_q3.json) |

### Why `--load-mode none` (`-lm none`), `-np 1`, and `-ub 512` Matter

1. **Fixing the `mmap` prompt-eval stall:** With default `mmap` and `-np 4` on Windows + Vulkan hybrid `-ot`, initial prompt processing stalled at **`17.48 tok/s`** (`65` tokens in [smoke_test_8k_q4kv.json](reports/smoke_test_8k_q4kv.json)). Switching to **`--load-mode none --parallel 1 --ubatch-size 512`** eliminated mmap paging stalls: prompt speed jumped **2.94×** on the tiny 65-token smoke test (`51.39 tok/s`) and reached **`151.88 tok/s` (`Q4`) / `159.72 tok/s` (`Q3`)** on full 2,046-token Pi system prompts (`226–246 tok/s` in `llama-bench -p 512 -ub 512`).
2. **Reading Windows Shared GPU Memory accurately:** With `-lm none`, `llama.cpp` allocates host-side `Vulkan_Host` buffers for the DDR4-offloaded experts (`8,974 MiB` on `Q4`, `~5.8 GB` total Shared DDR4 on `Q3`). Windows Task Manager / PDH telemetry reports that pinned host buffer under *Shared GPU Memory* (`~5.8 GB` Q3 / `~9.3 GB` Q4) — **that is the intentional DDR4 expert offload**, not GPU VRAM spillover, provided Dedicated VRAM stays under ~`10.8 GB`.

---

## 4. Observed Benchmarks & Live 32K `pi` Baseline

### Raw `llama-bench` (`q4_0`/`q4_0` KV, `-fa 1`, `-p 512 -n 128`, 3 reps)

| Model Quant | `-ub 256` Prompt (`pp512`) | `-ub 256` Gen (`tg128`) | `-ub 512` Prompt (`pp512`) | `-ub 512` Gen (`tg128`) | Report |
| --- | --- | --- | --- | --- | --- |
| **`UD-Q4_K_XL`** | `149.63 ± 5.07 t/s` | `14.13 ± 0.24 t/s` | **`226.68 ± 1.42 t/s`** | **`14.66 ± 0.04 t/s`** | [llama_bench_20261121_172954.txt](reports/llama_bench_20261121_172954.txt) |
| **`UD-Q3_K_XL`** | `186.72 ± 0.16 t/s` | `15.46 ± 0.05 t/s` | **`245.62 ± 0.82 t/s`** | **`15.43 ± 0.09 t/s`** | [llama_bench_20261121_173108.txt](reports/llama_bench_20261121_173108.txt) |

### Reproduction Run - 2026-11-21 21:35 Local (Idle GPU, No Server Resident)

Repeat of the exact committed sweep, with the live `llama-server` stopped and GPU memory sampled every 2 s. Pre-flight: port 9931 closed, no `llama*` process, idle dedicated VRAM `1,171 MB`, build still `b12615` (`a818a46f3`).

Weights were re-hashed before the run and are **identical** to the table in Section 2 (`e6427665...` Q4, `e2d03e8c...` Q3), so this is the same model, same build, same flags:

| Model Quant | `-ub 256` `pp512` | `-ub 256` `tg128` | `-ub 512` `pp512` | `-ub 512` `tg128` | Report |
| --- | --- | --- | --- | --- | --- |
| **`UD-Q4_K_XL`** (run 1) | `172.05 +/- 4.55 t/s` | `15.48 +/- 0.17 t/s` | `234.09 +/- 3.18 t/s` | `15.50 +/- 0.15 t/s` | [llama_bench_20261121_213537.txt](reports/llama_bench_20261121_213537.txt) |
| **`UD-Q4_K_XL`** (run 2) | `169.27 +/- 2.56 t/s` | `15.42 +/- 0.20 t/s` | `230.81 +/- 8.21 t/s` | `15.76 +/- 0.08 t/s` | [llama_bench_20261121_213909.txt](reports/llama_bench_20261121_213909.txt) |
| **`UD-Q3_K_XL`** | `193.24 +/- 0.41 t/s` | `16.20 +/- 0.09 t/s` | `250.75 +/- 4.91 t/s` | `16.16 +/- 0.24 t/s` | [llama_bench_20261121_213716.txt](reports/llama_bench_20261121_213716.txt) |

**Delta versus the original 17:29 / 17:31 baseline (same build, same weights):** every cell is faster, and the two independent Q4 runs agree within noise. Gain by cell (`ub 256` / `ub 512`): `pp` `+15.0%` / `+3.3%` (Q4), `+3.5%` / `+2.1%` (Q3); `tg` `+9.6%` / `+5.7%` (Q4), `+4.8%` / `+4.7%` (Q3).

- **`observed`:** the original baseline was measured on the same `b12615` build and the same GGUF bytes, so the gap is not a model or engine change. For clean-GPU runs, the honest top-line figure is **`15.4-16.2 tok/s` generation**, not `14-15`.
- **`hypothesis` (not verified):** the earlier run was slower because of GPU-side contention, most likely the live smoke-test `llama-server` still resident seconds before the `17:29:54` baseline run (last log line `17:29:25`), and/or Defender/disk activity on the freshly written `16.4 GiB` file. The old run cannot be re-tested after the fact.
- **Peak dedicated VRAM during the sweep:** `9,746 MB` (`Q4`) and `9,441 MB` (`Q3`) of `12,272 MB`; sampled *Shared* usage stayed at `~112 MB`, so **no WDDM spillover** occurred. Note these are `llama-bench` allocations (default context) and are not directly comparable to the `32K` server figures below.

### Depth Sweep - How the 32K Window Actually Behaves

One `llama-bench` invocation per quant with `-d 0,4096,8192,16384,32768`, `-ub 512`, `q4_0`/`q4_0`, `-fa 1`, `-p 512 -n 128`, `r=3`, loop-pinned `-ot`. This is the measurement the baseline was missing: every earlier number in this repo was depth `0`.

| Depth | Q4 `pp512` | Q4 `tg128` | Q3 `pp512` | Q3 `tg128` |
| --- | --- | --- | --- | --- |
| `0` | `221.35 +/- 1.97` | `14.69 +/- 0.13` | `250.68 +/- 2.64` | `16.31 +/- 0.04` |
| `4,096` | `121.87 +/- 0.45` | `14.76 +/- 0.04` | `128.20 +/- 0.25` | `15.89 +/- 0.20` |
| `8,192` | `99.40 +/- 0.25` | `14.60 +/- 0.03` | `101.68 +/- 0.31` | `15.15 +/- 0.05` |
| `16,384` | `72.66 +/- 0.22` | `14.23 +/- 0.02` | `74.19 +/- 0.17` | `14.87 +/- 0.04` |
| `32,768` | `41.96 +/- 0.12` | `14.09 +/- 0.09` | `43.63 +/- 0.10` | `14.21 +/- 0.02` |

Reports: [Q4](reports/llama_bench_20261121_215044.txt) / [Q3](reports/llama_bench_20261121_220739.txt).

- **Generation is nearly depth-invariant.** Q4 loses only `4.1%` (`14.69 -> 14.09`) across the full window, which matches the looped-expert cost model: per-token compute is dominated by the `12` looped layers, not by attention span. This is why long `pi` sessions feel steady once loaded.
- **Prompt processing collapses `5.3x`.** `221 -> 42` tok/s means re-reading `2,000` tokens at `32K` depth costs roughly `48 s` before the first output token. Prefix-cache reuse matters far more than raw `tg` for agent work.
- **Q3 degrades more at depth than Q4** (`16.31 -> 14.21`, `-12.9%` vs `-4.1%`). At `32K` the two quants are effectively tied at `~14.1 tok/s`, so Q3's `1.6 tok/s` advantage at depth `0` disappears exactly where long sessions live. Q3's real advantage is VRAM headroom, not speed.
- **Do not compare `-d 0` across run types.** In-list Q4 depth `0` reads `221.35` against `234.09` standalone (`~5%` gap), while Q3 is unchanged (`250.68` vs `250.75`). Passing a depth list makes `llama-bench` size the full `32K` context for every test, which shifts compute buffers for the larger quant. Only rows within one depth-list run are mutually comparable.

#### VRAM at 32K Is Over Target for Q4 (`observed`)

Sampled every 2 s during each sweep ([gpu_vram_depth_20261121_215044.csv](reports/gpu_vram_depth_20261121_215044.csv)); `12,272 MB` total dedicated:

| Sweep | Peak dedicated | Peak shared | vs `11,059 MB` target |
| --- | --- | --- | --- |
| Q4 (`-ub 512`) | `11,425 MB` | `2,416 MB` | **over by ~366 MB** |
| Q4 (`-ub 256`, `d32768` only) | `11,226 MB` | `701 MB` | **over by ~167 MB** |
| Q3 (`-ub 512`) | `10,826 MB` | `2,299 MB` | under by ~233 MB |

`-ub 256` at depth `32K` returns `pp512 44.45 +/- 0.09` and `tg128 13.69 +/- 0.03` ([report](reports/llama_bench_20261121_222347.txt)) - `pp` marginally better, `tg` marginally worse, and less spill.

**Verdict:** the `32K` server configuration reaches only `~10.2 GB` (Section 4) because it runs `-np 1` with its own buffer sizing, but `llama-bench` allocates the full depth for every test and pushes Q4 past the Windows headroom line at both ubatch sizes. Treat **Q4 `d32768` cells as indicative, not clean**: they ran at `~93%` dedicated VRAM with measurable spill, and no `-ub`/offload combination tried here brings them under `11,059 MB`. Q3 at `32K` stays inside the envelope and its depth curve should be the trusted one for long-session planning.

**8K `Vulkan0` buffer breakdown (`UD-Q4_K_XL`, `q4_0`/`q4_0`, `-np 1 -lm none -ub 512`):**
`7,827 MiB` model + `146 MiB` KV/RS context + `1,010 MiB` compute = **`8,984 MiB`** (`~9,969 MB` total dedicated VRAM including Windows desktop; full `-lv 4` breakdown in [smoke_test_8k_q4kv.json](reports/smoke_test_8k_q4kv.json)).

### Live 32K Multi-Turn `pi` Smoke Suite (`--thinking low`, `q4_0`/`q4_0` KV)

Both models were exercised back-to-back across all 3 standardized Windows diagnostics ([SMOKE_TEST_SUITE.md](SMOKE_TEST_SUITE.md)) in one continuous multi-turn `pi` session per quant (`--models-max 1` router dynamically evicting `Q3` and loading `Q4`):

| Model Quant | Dedicated VRAM (`12,288 MB`) | Test #1 (Hostname) | Test #2 (3 Unique Event Errors) | Test #3 (Hardware & Drivers) | Peak Tokens (`32K`) | Gen Speed | Truncations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`UD-Q3_K_XL`** | `9,586–9,593 MB` (`78.0%`, ~2.7 GB free) | `27.60s` *(cold load)* | `65.82s` | `92.61s` | `10,156 / 32,768` | `14.92–15.17 t/s` | `0` |
| **`UD-Q4_K_XL`** | `10,187–10,199 MB` (`82.9%`, ~2.1 GB free) | `27.49s` *(Q3→Q4 swap)* | `143.63s` *(self-fixed PS syntax)* | `124.86s` | `7,964 / 32,768` | `13.90–14.40 t/s` | `0` |

Full per-turn tool counts, `cacheRead` tokens, prompt/gen timings, and VRAM telemetry samples are recorded in [smoke_test_32k_q4kv.json](reports/smoke_test_32k_q4kv.json).

---

## 5. Pi & Windows Agent Gotchas Solved

1. **Do not mismatch `ctx-size` and Pi `contextWindow` (and don't run 8K for multi-turn tool sessions):**
   Pi reserves `4,096` safety tokens internally; on an `8K` server, an unmatched `contextWindow: 8192` leaves barely `~4K` before auto-compaction/clamp cuts responses mid-turn, while our sequential 3-prompt Windows diagnostic test actually hit `8,000–10,156` tokens. Expanding to **`ctx-size = 32768`** (`q4_0`/`q4_0` KV) and setting Pi **`contextWindow: 32768`** / **`maxTokens: 4096`** 1:1 leaves generous headroom (~`2.1 GB` GDDR6 still free even on `Q4`) and lets Pi auto-compact cleanly before touching the server limit.
2. **Explicitly map `thinkingLevelMap` + router reasoning caps:**
   Without `compat.supportsReasoningEffort` and `thinkingLevelMap` in `models.json`, `pi` fell back to high reasoning effort while `--reasoning-preserve` carried old `<think>` blocks forward. Setting `thinkingLevelMap` in [pi.models.example.json](configs/pi.models.example.json) alongside `--reasoning-effort low --reasoning-budget 1024 --no-reasoning-preserve` in [llama_router_presets.ini](configs/llama_router_presets.ini) keeps tool turns crisp. Notably, **`/thinking low` produced shorter `<think>` blocks than `/thinking off`** in multi-turn PowerShell sessions (`16–132` chars vs `473–815` chars earlier).
3. **Bound Windows CLI queries in prompts / `AGENTS.md`:**
   Raw `Get-WinEvent` and `Win32_PnPSignedDriver` dumps can flood smaller context windows, and offline models stall if asked to verify "latest online driver versions." Scoping diagnostics to `-MaxEvents`, explicit log names (`System`, `Application`), and local error codes (`Get-PnpDevice`, unsigned/oldest local drivers) — as in [SMOKE_TEST_SUITE.md](SMOKE_TEST_SUITE.md) — keeps outputs compact and grounded.

---

## 6. Quickstart Reproduction

1. **Upgrade `C:\llama.cpp` to mainline `b12615`** (backs up existing `C:\llama.cpp` to `C:\llama.cpp.bak_b12427`):
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/upgrade_llama_cpp.ps1
   ```
2. **Download & SHA-256 verify `UD-Q4_K_XL` (`Mainline_Support`) to `C:\models\Inkling-2-Mini-UD-Q4_K_XL.gguf`:**
   ```powershell
   python scripts/download_inkling.py
   ```
   *(Uses `HF_TOKEN` from environment or `~/.cache/huggingface/token` with safe CDN redirect auth stripping and 12 parallel HTTP Range workers.)*
3. **Verify the exact `-ot` VRAM/DDR4 tensor split before launching:**
   ```powershell
   python scripts/inspect_inkling_tensors.py
   ```
4. **Launch either the router (recommended for `pi`) or standalone server:**
   - Router mode with automatic `Q4` / `Q3` swapping:
     ```powershell
     C:\llama.cpp\llama.exe serve --models-dir C:\models --models-preset configs\llama_router_presets.ini --models-max 1 --host 127.0.0.1 --port 9931
     ```
   - Standalone single-model server (`8088`, 32K `q4_0`/`q4_0` defaults):
     ```powershell
     powershell -NoProfile -ExecutionPolicy Bypass -File scripts/launch_inkling.ps1
     ```
5. **Configure `pi` (`~/.pi/agent/models.json`):** Copy the `"local-llama-router"` block from [configs/pi.models.example.json](configs/pi.models.example.json), then run `pi --thinking low`.
6. **Optional synthetic benchmark sweep (`llama-bench`):**
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bench_inkling.ps1
   ```

---

## 7. Repository Map

| Path | Purpose |
| --- | --- |
| [configs/llama_router_presets.ini](configs/llama_router_presets.ini) | Verified `llama.cpp` router presets for 32K `q4_0`/`q4_0` KV + loop-pinned `-ot` |
| [configs/pi.models.example.json](configs/pi.models.example.json) | Sanitized `pi` `models.json` template (`1:1` `32768`/`4096` + `thinkingLevelMap`) |
| [SMOKE_TEST_SUITE.md](SMOKE_TEST_SUITE.md) | Standardized 3-prompt Windows diagnostic smoke suite & Pass/Fail rubric |
| [scripts/upgrade_llama_cpp.ps1](scripts/upgrade_llama_cpp.ps1) | Mainline `b12615` Vulkan x64 installer with rollback backup |
| [scripts/download_inkling.py](scripts/download_inkling.py) | Resumable 12-worker HTTP Range downloader & SHA-256 verifier |
| [scripts/inspect_inkling_tensors.py](scripts/inspect_inkling_tensors.py) | GGUF metadata + `-ot` VRAM/DDR4 tensor byte splitter |
| [scripts/launch_inkling.ps1](scripts/launch_inkling.ps1) | Standalone `llama-server` launcher aligned to 32K `q4_0`/`q4_0` baseline |
| [scripts/bench_inkling.ps1](scripts/bench_inkling.ps1) | `llama-bench` `-ub 256,512` sweep runner |
| [reports/smoke_test_32k_q4kv.json](reports/smoke_test_32k_q4kv.json) | Observed 32K `pi` multi-turn smoke results (`Q3` & `Q4`) + VRAM telemetry |
| [reports/smoke_test_8k_q4kv.json](reports/smoke_test_8k_q4kv.json) | Earlier 8K `mmap` stall vs `-lm none` fix + full `-lv 4` `Vulkan0` breakdown |
| [reports/llama_bench_20261121_172954.txt](reports/llama_bench_20261121_172954.txt) | `llama-bench` output — `UD-Q4_K_XL` (`-ub 256,512`) |
| [reports/llama_bench_20261121_173108.txt](reports/llama_bench_20261121_173108.txt) | `llama-bench` output — `UD-Q3_K_XL` (`-ub 256,512`) |
| [reports/tensor_layout.json](reports/tensor_layout.json) | Verified GGUF tensor split (`UD-Q4_K_XL`) |
| [reports/tensor_layout_q3.json](reports/tensor_layout_q3.json) | Verified GGUF tensor split (`UD-Q3_K_XL`) |
| [reports/llama_bench_20261121_213537.txt](reports/llama_bench_20261121_213537.txt) | Clean-GPU reproduction - `UD-Q4_K_XL` (run 1) |
| [reports/llama_bench_20261121_213909.txt](reports/llama_bench_20261121_213909.txt) | Clean-GPU reproduction - `UD-Q4_K_XL` (run 2, stability check) |
| [reports/llama_bench_20261121_213716.txt](reports/llama_bench_20261121_213716.txt) | Clean-GPU reproduction - `UD-Q3_K_XL` |
| [reports/gpu_vram_20261121_213530.csv](reports/gpu_vram_20261121_213530.csv) | 2 s dedicated/shared VRAM samples across the reproduction sweep |
| [reports/llama_bench_20261121_215044.txt](reports/llama_bench_20261121_215044.txt) | Depth sweep - `UD-Q4_K_XL`, `-d 0,4096,8192,16384,32768` |
| [reports/llama_bench_20261121_220739.txt](reports/llama_bench_20261121_220739.txt) | Depth sweep - `UD-Q3_K_XL`, `-d 0,4096,8192,16384,32768` |
| [reports/llama_bench_20261121_222347.txt](reports/llama_bench_20261121_222347.txt) | `UD-Q4_K_XL` at `d32768` with `-ub 256` (VRAM recheck) |
| [reports/gpu_vram_depth_20261121_215044.csv](reports/gpu_vram_depth_20261121_215044.csv) | 2 s VRAM samples across all depth runs (851 rows) |
