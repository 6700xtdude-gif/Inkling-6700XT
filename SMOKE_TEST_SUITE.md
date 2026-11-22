# Standardized Windows Pi Smoke Test Suite (Small-Model Friendly)

A progressive 3-prompt manual smoke test suite designed for local 4B–27B models running in `pi` on Windows (`3D2A-STUDIO` or `SANDBOX`). Each prompt keeps the diagnostic task realistic while bounding raw Windows WMI/Event Log output so smaller models don't exhaust context or stall trying to perform impossible online driver lookups.

## Recommended Run Settings
- **Thinking level:** `/thinking low` (or `--thinking low`) on Inkling 2 Mini — observed shorter, cleaner reasoning than `/thinking off`.
- **Session mode:** Start fresh with `/new` to score each prompt in isolation, **or** run all 3 back-to-back in one session at **16K–32K+ context** to verify multi-turn KV caching (`8K` is too small once Test #2 + #3 run sequentially).
- **PowerShell 5.1 tip:** If a model ever reports empty stdout from `Select-Object`, remind it (or keep in `AGENTS.md`) to pipe tables through `Format-Table -AutoSize | Out-String` or `ConvertTo-Json`.

---

## Test #1 — Tool Grounding & Strict Output (Hostname)

```text
Using PowerShell, check the exact computer name (hostname) of this Windows machine and reply with only the hostname.
```

- **What it tests:** Calls the PowerShell tool instead of guessing, and respects a strict single-line output constraint.
- **Pass criteria:** `1` tool call (`hostname` / `$env:COMPUTERNAME`), returns exact hostname (`3D2A-STUDIO` or `SANDBOX`), no extra commentary.

---

## Test #2 — Scoped Querying, Deduplication & Formatting (Event Log)

```text
Using PowerShell, inspect up to the last 50 Error-level events (Level 2) from the Windows System and Application event logs only. List the 3 most recent unique error messages with: Timestamp, Log, Source, Event ID, and a 1-sentence summary of each. Keep command output short so it doesn't flood context.
```

- **What it tests:** Scopes away permission-denied logs (e.g., `Security`), bounds `Get-WinEvent` with `-MaxEvents` / hashtable filters, deduplicates repeated crashes, and formats concise output.
- **Pass criteria:** No full-log hang/flood; identifies `3` distinct recent errors with Timestamp, Log, Source, Event ID, and accurate 1-sentence summary.

---

## Test #3 — Local Hardware & Driver Health (Bounded & Honest Scope)

```text
Using PowerShell, run a concise local hardware and driver health check on this PC: (1) list the CPU, GPU(s), and total RAM; (2) report any Plug-and-Play devices with a non-OK status or missing/problem driver (Get-PnpDevice where Status is not OK); (3) list the 5 oldest or unsigned installed drivers (DeviceName, DriverVersion, DriverDate, IsSigned) from Win32_PnPSignedDriver. Cap command output to keep context small, and do not guess online latest driver versions.
```

- **What it tests:** Multi-part local CIM/WMI + PnP inspection, table capping (`Select-Object -First`), and epistemic honesty (never fabricating "latest available vendor versions" offline).
- **Pass criteria:** Accurately reports CPU/GPU/RAM; flags real `Error` PnP devices cleanly separately from harmless `Unknown`/`CM_PROB_PHANTOM` entries; checks `Win32_PnPSignedDriver` cleanly without guessing missing fields or online versions.
  *(Tip for future prompt tuning: on Windows PCs where virtual/classless entries have blank `DriverDate`, `Win32_PnPSignedDriver | Where-Object { $_.DriverDate }` avoids sorting blank dates first.)*

---

## Observed 32K `q4_0`/`q4_0` KV Baseline (`Inkling-2-Mini`, `--thinking low`)

Tested back-to-back in a single multi-turn `pi` session per quant from `3D2A-STUDIO` against SANDBOX router (`http://10.10.10.1:9931`, `llama.cpp b12615`, loop-pinned `-ot`, `ctx-size = 32768`, `q4_0`/`q4_0` KV, `ContextWindow = 32768`). Raw metrics: [smoke_test_32k_q4kv.json](reports/smoke_test_32k_q4kv.json).

| Model Quant | Dedicated VRAM (SANDBOX 6700 XT) | Test #1 Time | Test #2 Time | Test #3 Time | Peak Session Tokens | Gen Speed (`tok/s`) | Truncations | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`UD-Q3_K_XL`** | `9,586–9,593 MB` (`78.0%`, ~2.7 GB free) | `27.60s` *(cold load)* | `65.82s` | `92.61s` | `10,156 / 32,768` | `14.92–15.17` | `0` | **3/3 PASS** |
| **`UD-Q4_K_XL`** | `10,187–10,199 MB` (`82.9%`, ~2.1 GB free) | `27.49s` *(hot swap from Q3)* | `143.63s` | `124.86s` | `7,964 / 32,768` | `13.90–14.40` | `0` | **3/3 PASS** |

*Notes:*
- `Shared DDR4` (`~5.8 GB` Q3 / `~9.3 GB` Q4) reflects intentional `-lm none` `Vulkan_Host` DDR4 offload, not GPU spill.
- Dedicated VRAM on `UD-Q4_K_XL` stayed safely below the ~`10.8 GB` Windows WDDM ceiling even at `32K` context.
- During `UD-Q4_K_XL` Test #2, two initial `Get-WinEvent` calls hit a PowerShell parameter-set syntax error and the model self-corrected on the next turn without user intervention.
