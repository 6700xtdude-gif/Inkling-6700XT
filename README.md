# Inkling-6700XT

Deployment, tensor-layout verification, and benchmarking workspace for running **Thinking Machines Lab's Inkling 2 Mini** (`unsloth/Inkling-2-Mini-GGUF`, `Mainline_Support` `UD-Q4_K_XL`) on **SANDBOX**:

- **GPU:** AMD Radeon RX 6700 XT 12 GB (`gfx1031`, Vulkan)
- **CPU:** AMD Ryzen 5 5600 (6C/12T)
- **RAM:** 64 GB DDR4-3200
- **OS:** Windows 11 Pro (keep dedicated GPU VRAM under ~10.8 GB to avoid silent WDDM spillover into DDR4)

## Why `Mainline_Support` + `llama.cpp b12615`

`unsloth/Inkling-2-Mini-GGUF` publishes two `UD-Q4_K_XL` files (both 17.63 GB / 16.42 GiB):
- Root `Inkling-2-Mini-UD-Q4_K_XL.gguf` targets Unsloth's custom fork.
- `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file` targets upstream `ggml-org/llama.cpp` (`inkling2_loop` architecture).

Official mainline **`llama.cpp b12615`** includes both `#32694` (`inkling2_loop` architecture) and `#32704` (per-expert NVFP4 scale fix across loop passes), allowing a clean drop-in upgrade of `C:\llama.cpp` from `b12427` while preserving compatibility with the existing `llama.exe` router (`C:\models`) and `ENDO` tools.

## Initial Launch Profile (8K Context + `q4_0`/`q4_0` KV)

Inkling 2 Mini has 24 physical layers where **middle layers `6..17` run 3x per token** and outer layers `0..5` and `18..23` run 1x per token. Instead of naive `--n-cpu-moe` (which evicts early looped layers to DDR4), we pin all dense/attention/looped layers (`6..17`) in VRAM and offload only the 1x outer MoE expert tensors (`0..5`, `18..23`) to 64 GB DDR4:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/launch_inkling.ps1
```

Default flags:
- `-ngl 99`
- `-ot "blk\.([0-5]|1[8-9]|2[0-3])\.ffn_.*_exps\.=CPU"`
- `-c 8192 -fa on --cache-type-k q4_0 --cache-type-v q4_0`
- No vision `mmproj` loaded (maximizes VRAM headroom for context & compute buffers)

Once VRAM headroom is confirmed with `vram_telemetry.ps1` (Dedicated VRAM < 10.8 GB, Shared DDR4 near 0 MB), step up KV cache (`-CacheTypeK q8_0 -CacheTypeV q8_0`) or context (`-Ctx 16384` / `-Ctx 32768`).

## Scripts

- `scripts/download_inkling.py`: Resumable `curl.exe` download of `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file` (`17,629,760,736` bytes, SHA-256 verified) to `C:\models\Inkling-2-Mini-UD-Q4_K_XL.gguf`.
- `scripts/upgrade_llama_cpp.ps1`: Backs up `C:\llama.cpp` to `C:\llama.cpp.bak_b12427`, installs mainline `b12615` (`win-vulkan-x64`), preserves local helpers (`start-qwen7b.*`, `llama-connect.exe`), and verifies `llama.exe` + RX 6700 XT Vulkan detection.
- `scripts/inspect_inkling_tensors.py`: Parses GGUF metadata and tensor byte sizes (works on `.gguf` or active `.gguf.part`) and verifies the exact `-ot` VRAM/DDR4 split.
- `scripts/launch_inkling.ps1`: Starts `llama-server.exe` with the 8K `q4_0`/`q4_0` loop-pinned profile.
- `scripts/bench_inkling.ps1`: Runs `llama-bench.exe` (`-p 512 -n 128`) and saves results under `reports/`.
