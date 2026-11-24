# Generated summary - INKLING-CONFIRM-001

- Generated: 2026-11-23T18:30:10.369293-06:00
- Spec: `TEST_SPEC.md`
- Spec SHA-256: `7F64470190E78106910AB51AB90F02BD01334D904CB21AA0B97309D7C19DCBC5`
- Engine: llama-bench / build a818a46f3 (12615)

## Raw sources

| File | Role | Quant | SHA-256 |
|---|---|---|---|
| reports/llama_bench_20261121_213537.txt | baseline | Q4 | `A38FFE551B5735888B08D141CF152597DCCAB87E33CA0D2695429BF2EF98C2CB` |
| reports/llama_bench_20261121_213716.txt | baseline | Q3 | `41B5606250734763FB4A83A3B1855E5ECDF1A4C44403D648937BA79459D8C9EF` |
| reports/llama_bench_confirm_20261122_000959.txt | confirm |  | `6286B5EA5D7905E8BFE716F2983F4F2086EDB9CBA24A27A90E9D6DDBBBF50AB5` |
| reports/llama_bench_20261121_215044.txt | depth | Q4 | `F30E94139618AF6DFAF30DE546AB0069CAFC22C04AE2E463435CD9FD19024ABC` |
| reports/llama_bench_20261121_220739.txt | depth | Q3 | `44DBA0C0DEE0CF0FF071806B0A8AA845E9CA29F791C4E81D9BD50D783C1B6744` |
| reports/gpu_vram_20261121_213530.csv | vram |  | `CD6C0524CA762FEAE07FCDC37B4C88D7BC92FDEDDAD2CFD5020B699D0AEA6516` |
| reports/llama_bench_confirm_20261122_000922.txt | aborted |  | `C6E97FD0471D61E9E39E065030021E3706973234630D4E69115FEE59DEADABB7` |

## Parsed values

| Role | Quant | ubatch | depth | metric | value | +/- |
|---|---|---:|---:|---|---:|---:|
| baseline | Q4 | 256 | 0 | pp512 | 172.05 | 4.55 |
| baseline | Q4 | 256 | 0 | tg128 | 15.48 | 0.17 |
| baseline | Q4 | 512 | 0 | pp512 | 234.09 | 3.18 |
| baseline | Q4 | 512 | 0 | tg128 | 15.50 | 0.15 |
| baseline | Q3 | 256 | 0 | pp512 | 193.24 | 0.41 |
| baseline | Q3 | 256 | 0 | tg128 | 16.20 | 0.09 |
| baseline | Q3 | 512 | 0 | pp512 | 250.75 | 4.91 |
| baseline | Q3 | 512 | 0 | tg128 | 16.16 | 0.24 |
| confirm | Q4 | 256 | 0 | pp512 | 167.85 | 4.17 |
| confirm | Q4 | 256 | 0 | tg128 | 15.07 | 0.08 |
| confirm | Q4 | 512 | 0 | pp512 | 228.99 | 1.57 |
| confirm | Q4 | 512 | 0 | tg128 | 15.09 | 0.04 |
| confirm | Q3 | 256 | 0 | pp512 | 191.12 | 0.59 |
| confirm | Q3 | 256 | 0 | tg128 | 15.68 | 0.02 |
| confirm | Q3 | 512 | 0 | pp512 | 248.12 | 1.94 |
| confirm | Q3 | 512 | 0 | tg128 | 15.63 | 0.01 |
| depth | Q4 | 0 | 0 | pp512 | 221.35 | 1.97 |
| depth | Q4 | 0 | 0 | tg128 | 14.69 | 0.13 |
| depth | Q4 | 0 | 4096 | pp512 | 121.87 | 0.45 |
| depth | Q4 | 0 | 4096 | tg128 | 14.76 | 0.04 |
| depth | Q4 | 0 | 8192 | pp512 | 99.40 | 0.25 |
| depth | Q4 | 0 | 8192 | tg128 | 14.60 | 0.03 |
| depth | Q4 | 0 | 16384 | pp512 | 72.66 | 0.22 |
| depth | Q4 | 0 | 16384 | tg128 | 14.23 | 0.02 |
| depth | Q4 | 0 | 32768 | pp512 | 41.96 | 0.12 |
| depth | Q4 | 0 | 32768 | tg128 | 14.09 | 0.09 |
| depth | Q3 | 0 | 0 | pp512 | 250.68 | 2.64 |
| depth | Q3 | 0 | 0 | tg128 | 16.31 | 0.04 |
| depth | Q3 | 0 | 4096 | pp512 | 128.20 | 0.25 |
| depth | Q3 | 0 | 4096 | tg128 | 15.89 | 0.20 |
| depth | Q3 | 0 | 8192 | pp512 | 101.68 | 0.31 |
| depth | Q3 | 0 | 8192 | tg128 | 15.15 | 0.05 |
| depth | Q3 | 0 | 16384 | pp512 | 74.19 | 0.17 |
| depth | Q3 | 0 | 16384 | tg128 | 14.87 | 0.04 |
| depth | Q3 | 0 | 32768 | pp512 | 43.63 | 0.10 |
| depth | Q3 | 0 | 32768 | tg128 | 14.21 | 0.02 |

## Deltas (calculated, no verdict)

- tg128 ub256: Q3 15.68 +/- 0.02 -> Q4 15.07 +/- 0.08 (delta -0.61, -3.9%)
- tg128 ub512: Q3 15.63 +/- 0.01 -> Q4 15.09 +/- 0.04 (delta -0.54, -3.5%)
- pp512 ub256: Q3 191.12 +/- 0.59 -> Q4 167.85 +/- 4.17 (delta -23.27, -12.2%)
- pp512 ub512: Q3 248.12 +/- 1.94 -> Q4 228.99 +/- 1.57 (delta -19.13, -7.7%)

## VRAM peaks

- Q3: 9,441 MB
- Q4: 9,746 MB

## Labels

- Values and deltas: `observed` (derived from raw artifacts).
- Cause of any difference: `hypothesis` until tested.
- No automatic pass/fail verdict is issued.
