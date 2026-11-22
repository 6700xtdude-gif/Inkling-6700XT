# Inkling 2 Mini on AMD Radeon RX 6700 XT (12 GB VRAM + 64 GB DDR4)

**Verdict: Thumbs-up baseline — functional with `pi` coding agent, acceptably slow at ~14–15 tok/s, zero truncations at 32K.**

This repository documents the exact Windows 11 Vulkan setup, loop-aware `-ot` tensor offload strategy, `llama.cpp` router preset, `pi` client config, benchmarks, and live multi-turn agent smoke tests for running Thinking Machines Lab's **Inkling 2 Mini** (`unsloth/Inkling-2-Mini-GGUF`, `Mainline_Support` `UD-Q4_K_XL` and `UD-Q3_K_XL`) on a **12 GB AMD Radeon RX 6700 XT**.

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
