---
spec_id: INKLING-CONFIRM-001
version: 1
created: 2026-11-22
author: 6700XT
repo_root: C:/Users/6700XT/.gemini/antigravity/scratch/Inkling-6700XT
question: Does Q3 remain faster and lighter than Q4 after the 22:37 binary swap?
hypothesis: Q3 remains faster and lighter than Q4 on this rig.
engine: llama-bench
engine_exe: C:/llama.cpp/llama-bench.exe
engine_sha256: D9AADBA13D0EA58330C45AA0F96A107F75E06B037F58D06745C35DF4B918C6C3
build_string: a818a46f3 (12615)
model_1_label: Q4
model_1_path: C:/models/Inkling-2-Mini-UD-Q4_K_XL.gguf
model_1_sha256: E64276656F072E9B2197A61723667CBDB71DA66A716FFD966E572C21088F3CD3
model_2_label: Q3
model_2_path: C:/models/Inkling-2-Mini-UD-Q3_K_XL.gguf
model_2_sha256: E2D03E8C92AC21CFF0E3309B95F40DFD27D1825184F0D2BC09DB249797273639
flags: -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -p 512 -n 128
verbatim_command: "& 'C:\\llama.cpp\\llama-bench.exe' -m '<model>' -ngl 99 -ot 'blk\\.([0-5]|1[8-9]|2[0-3])\\.ffn_.*_exps\\.=CPU' -fa 1 -ctk q4_0 -ctv q4_0 -ub '256,512' -p 512 -n 128 -r 3"
override_tensor: blk\.([0-5]|1[8-9]|2[0-3])\.ffn_.*_exps\.=CPU
reps: 3
ubatch: [256, 512]
depth: [0]
metrics: [pp512, tg128]
preflight: [gpu_idle, router_stopped, hashes_match, vram_baseline]
postflight: [restore_router, verify_bind, exit_zero]
router_exe: C:/llama.cpp/llama.exe
router_args: [serve, --models-dir, C:/models, --models-preset, configs/llama_router_presets.ini, --models-max, 1, --host, 10.10.10.1, --port, 9931]
router_host: 10.10.10.1
router_port: 9931
router_cwd: C:/Users/6700XT/.gemini/antigravity/scratch/Inkling-6700XT
vram_command: ""
output_dir: specs/INKLING-CONFIRM-001/raw
source_1_file: reports/llama_bench_20261121_213537.txt
source_1_role: baseline
source_1_quant: Q4
source_2_file: reports/llama_bench_20261121_213716.txt
source_2_role: baseline
source_2_quant: Q3
source_3_file: reports/llama_bench_confirm_20261122_000959.txt
source_3_role: confirm
source_4_file: reports/llama_bench_20261121_215044.txt
source_4_role: depth
source_4_quant: Q4
source_5_file: reports/llama_bench_20261121_220739.txt
source_5_role: depth
source_5_quant: Q3
source_6_file: reports/gpu_vram_20261121_213530.csv
source_6_role: vram
source_6_labels: [Q4, Q3]
source_6_split_time: 21:37:25
source_7_file: reports/llama_bench_confirm_20261122_000922.txt
source_7_role: aborted
---

# Question

The binaries were replaced at 22:37 on 2026-11-21, after the 22:33 benchmark run.
The README claimed Q3 was slightly faster and lighter than Q4. This pilot re-expresses
the 2026-11-22 confirmation run that tested whether that ranking still held, using only
raw artifacts that already exist. No GPU work is performed by this spec.

# Hypothesis

Q3 remains faster and lighter than Q4. `hypothesis` until the measurement below was
taken; the raw confirmation artifact now supports it as `observed` for these exact flags.

# Matrix

- Models: Q4 and Q3 `Mainline_Support` GGUFs, hashes recorded above.
- Shared flags: `-ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -p 512 -n 128`.
- Loop-pin override `-ot` applied to both.
- Original confirmation artifact ran one invocation per model with `-ub '256,512'` (see `verbatim_command` above); `run_spec.py` invokes `-ub 256` and `-ub 512` separately, with `-r 3` and no `-d` flag (depth 0).
- Depth sweep sources used `-ub 512` with `-d 0,4096,8192,16384,32768`.
- One lever at a time: quant is the only changed model axis in the comparison.

# Preflight

- Model SHA-256 values matched the README before the GPU was touched.
- Engine and server SHA-256 recorded in the confirmation artifact.
- Router stopped for a clean VRAM window, then restored.
- Build string `a818a46f3 (12615)` confirmed in the raw output.

# Postflight

- Router restored on `10.10.10.1:9931`, PID 2680.
- Listener verified with `Get-NetTCPConnection`.
- Both benchmark attempts exited 0.

# Decision log

- First attempt, `llama_bench_confirm_20261122_000922.txt`, was aborted: PowerShell
  treated normal llama-bench stderr as a terminating NativeCommandError. Kept on disk
  with `STATUS: ABORTED`. Source 7 records it; it contributes no metric rows.
- An earlier encoding failure (mangled em dash in a `.ps1`) was also kept as data.
- README line 167 still carries a warmup VRAM figure with no raw telemetry. Not fixed
  by this spec; the human gate applies.
- Absolute numbers came in 1-3% below the 11-21 baseline. Cause untested: `hypothesis`.
