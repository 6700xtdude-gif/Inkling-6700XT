"""
Inspect GGUF v2/v3 metadata, exact tensor names, per-tensor byte sizes,
and simulate -ot regex GPU-vs-CPU VRAM splits (works on .gguf or active .gguf.part).
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

GGUF_MAGIC = 0x46554747  # 'GGUF'

DEFAULT_MODEL = Path(r"C:\models\Inkling-2-Mini-UD-Q4_K_XL.gguf")
DEFAULT_PART = Path(r"C:\models\Inkling-2-Mini-UD-Q4_K_XL.gguf.part")
DEFAULT_OT_REGEX = r"blk\.([0-5]|1[8-9]|2[0-3])\.ffn_.*_exps\."

GGUF_VALUE_SIZES = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
}

# (block_elements, block_bytes) for exact fallback sizing on the final tensor
GGML_QUANT_INFO = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
    8: (32, 34), 9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74),
    18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82),
    23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66), 39: (32, 17),
}


def read_str(f) -> str:
    n_bytes = f.read(8)
    if len(n_bytes) < 8:
        raise EOFError("Unexpected EOF reading GGUF string length")
    n = struct.unpack("<Q", n_bytes)[0]
    raw = f.read(n)
    if len(raw) < n:
        raise EOFError("Unexpected EOF reading GGUF string bytes")
    return raw.decode("utf-8", errors="replace")


def read_scalar_or_str(f, vtype: int):
    if vtype == 0:
        return struct.unpack("<B", f.read(1))[0]
    if vtype == 1:
        return struct.unpack("<b", f.read(1))[0]
    if vtype == 2:
        return struct.unpack("<H", f.read(2))[0]
    if vtype == 3:
        return struct.unpack("<h", f.read(2))[0]
    if vtype == 4:
        return struct.unpack("<I", f.read(4))[0]
    if vtype == 5:
        return struct.unpack("<i", f.read(4))[0]
    if vtype == 6:
        return struct.unpack("<f", f.read(4))[0]
    if vtype == 7:
        return bool(struct.unpack("<B", f.read(1))[0])
    if vtype == 8:
        return read_str(f)
    if vtype == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11:
        return struct.unpack("<q", f.read(8))[0]
    if vtype == 12:
        return struct.unpack("<d", f.read(8))[0]
    if vtype == 9:
        atype = struct.unpack("<I", f.read(4))[0]
        alen = struct.unpack("<Q", f.read(8))[0]
        if atype in GGUF_VALUE_SIZES:
            f.seek(GGUF_VALUE_SIZES[atype] * alen, 1)
            return {"array_type": atype, "count": alen}
        if atype == 8:
            sample = []
            for idx in range(alen):
                s = read_str(f)
                if idx < 4:
                    sample.append(s)
            return {"array_type": 8, "count": alen, "sample": sample}
        if atype == 9:
            for _ in range(alen):
                read_scalar_or_str(f, 9)
            return {"array_type": 9, "count": alen}
        raise ValueError(f"Unsupported nested GGUF array element type: {atype}")
    raise ValueError(f"Unsupported GGUF metadata type: {vtype}")


def estimate_tensor_bytes(dims: list[int], ggml_type: int) -> int:
    nelem = 1
    for d in dims:
        nelem *= d
    if ggml_type in GGML_QUANT_INFO:
        blk_elems, blk_bytes = GGML_QUANT_INFO[ggml_type]
        return ((nelem + blk_elems - 1) // blk_elems) * blk_bytes
    return max(0, nelem // 2)


def parse_gguf_layout(path: Path) -> tuple[int, dict, list[dict], int]:
    file_bytes = path.stat().st_size
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file ({path}): magic=0x{magic:08x}")
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]

        metadata: dict = {}
        for _ in range(kv_count):
            key = read_str(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            metadata[key] = read_scalar_or_str(f, vtype)

        tensors: list[dict] = []
        for _ in range(tensor_count):
            name = read_str(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ggml_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors.append({
                "name": name,
                "dims": dims,
                "ggml_type": ggml_type,
                "type_name": GGML_TYPE_NAMES.get(ggml_type, f"TYPE_{ggml_type}"),
                "offset": offset,
            })

        header_end = f.tell()

    alignment = int(metadata.get("general.alignment", 32) or 32)
    data_start = ((header_end + alignment - 1) // alignment) * alignment

    # Exact per-tensor byte sizes from sorted offsets (last tensor via quant formula or EOF)
    by_off = sorted(range(len(tensors)), key=lambda i: tensors[i]["offset"])
    for pos, idx in enumerate(by_off):
        t = tensors[idx]
        if pos + 1 < len(by_off):
            nxt_off = tensors[by_off[pos + 1]]["offset"]
            t["bytes"] = max(0, nxt_off - t["offset"])
        else:
            est = estimate_tensor_bytes(t["dims"], t["ggml_type"])
            if file_bytes > data_start + t["offset"]:
                eof_rem = file_bytes - (data_start + t["offset"])
                t["bytes"] = min(est, eof_rem) if est > 0 else eof_rem
            else:
                t["bytes"] = est

    return version, metadata, tensors, data_start


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect GGUF metadata, tensor names, and -ot VRAM/CPU split.")
    parser.add_argument("--path", default=None, help="Path to .gguf or .gguf.part (auto-detects Inkling-2-Mini).")
    parser.add_argument("--ot-regex", default=DEFAULT_OT_REGEX, help="Regex for CPU offload tensor override (-ot).")
    parser.add_argument("--json-out", default=None, help="Optional path to save layout summary JSON.")
    args = parser.parse_args()

    if args.path:
        target = Path(args.path)
    elif DEFAULT_MODEL.is_file():
        target = DEFAULT_MODEL
    elif DEFAULT_PART.is_file():
        target = DEFAULT_PART
    else:
        print(f"[ERROR] Neither {DEFAULT_MODEL} nor {DEFAULT_PART} exists.", file=sys.stderr)
        return 1

    print(f"Inspecting GGUF header: {target} (on-disk: {target.stat().st_size / 1024**3:.2f} GiB)")
    version, metadata, tensors, data_start = parse_gguf_layout(target)
    arch = str(metadata.get("general.architecture", "unknown"))

    print(f"GGUF v{version} | tensors={len(tensors)} | kv_keys={len(metadata)} | data_start={data_start:,} B")
    print("\n=== Core Metadata ===")
    interesting_tokens = ("general.", arch + ".", "loop", "layer", "expert", "context", "attention", "head", "rope")
    for k in sorted(metadata):
        if any(tok in k.lower() for tok in interesting_tokens):
            v = metadata[k]
            if isinstance(v, dict) and "count" in v and "array_type" in v:
                continue
            print(f"  {k}: {v}")

    pattern_stats: dict[str, dict] = defaultdict(lambda: {"count": 0, "bytes": 0, "types": set(), "sample_dims": None})
    layer_bytes: dict[int, int] = defaultdict(int)
    layer_exps_bytes: dict[int, int] = defaultdict(int)
    non_layer_bytes = 0

    for t in tensors:
        name = t["name"]
        b = t["bytes"]
        norm = re.sub(r"\bblk\.\d+\.", "blk.<N>.", name)
        st = pattern_stats[norm]
        st["count"] += 1
        st["bytes"] += b
        st["types"].add(t["type_name"])
        if st["sample_dims"] is None:
            st["sample_dims"] = t["dims"]

        m = re.match(r"^blk\.(\d+)\.", name)
        if m:
            blk_idx = int(m.group(1))
            layer_bytes[blk_idx] += b
            if "_exps." in name:
                layer_exps_bytes[blk_idx] += b
        else:
            non_layer_bytes += b

    print("\n=== Distinct Tensor Name Patterns ===")
    for pat in sorted(pattern_stats):
        st = pattern_stats[pat]
        tlist = ",".join(sorted(st["types"]))
        print(
            f"  {pat:<38} | cnt={st['count']:>3} | {st['bytes'] / 1024**2:8.1f} MiB "
            f"| {tlist:<12} | dims={st['sample_dims']}"
        )

    rx = re.compile(args.ot_regex)
    cpu_tensors = [t for t in tensors if rx.search(t["name"])]
    gpu_tensors = [t for t in tensors if not rx.search(t["name"])]
    cpu_bytes = sum(t["bytes"] for t in cpu_tensors)
    gpu_bytes = sum(t["bytes"] for t in gpu_tensors)
    total_bytes = cpu_bytes + gpu_bytes

    print(f"\n=== Simulated -ot Split: `{args.ot_regex}=CPU` ===")
    print(f"  Total tensor payload: {total_bytes / 1024**3:6.2f} GiB ({total_bytes / 1e9:6.2f} GB)")
    print(f"  Pinned in GPU VRAM:   {gpu_bytes / 1024**3:6.2f} GiB ({gpu_bytes / 1e9:6.2f} GB) across {len(gpu_tensors)} tensors")
    print(f"  Offloaded to DDR4:    {cpu_bytes / 1024**3:6.2f} GiB ({cpu_bytes / 1e9:6.2f} GB) across {len(cpu_tensors)} tensors")
    print(f"  Non-layer tensors:    {non_layer_bytes / 1024**2:6.1f} MiB")

    print("\n=== Per-Layer Breakdown (MiB) ===")
    for blk in sorted(layer_bytes):
        tot_mib = layer_bytes[blk] / 1024**2
        exp_mib = layer_exps_bytes[blk] / 1024**2
        dense_mib = tot_mib - exp_mib
        cpu_tag = "CPU_EXPS" if any(rx.search(f"blk.{blk}.ffn_gate_exps.weight") or rx.search(t["name"]) for t in cpu_tensors if t["name"].startswith(f"blk.{blk}.")) else "ALL_VRAM"
        print(f"  blk.{blk:02d} | total={tot_mib:7.1f} MiB (dense/attn={dense_mib:6.1f}, exps={exp_mib:6.1f}) | {cpu_tag}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "inspected_file": str(target),
            "gguf_version": version,
            "architecture": arch,
            "tensor_count": len(tensors),
            "ot_regex": args.ot_regex,
            "total_tensor_bytes": total_bytes,
            "gpu_vram_weight_bytes": gpu_bytes,
            "cpu_ddr4_weight_bytes": cpu_bytes,
            "gpu_tensor_count": len(gpu_tensors),
            "cpu_tensor_count": len(cpu_tensors),
            "metadata_summary": {
                k: v for k, v in sorted(metadata.items())
                if any(tok in k.lower() for tok in interesting_tokens)
                and not (isinstance(v, dict) and "count" in v and "array_type" in v)
            },
            "patterns": {
                pat: {
                    "count": st["count"],
                    "bytes": st["bytes"],
                    "types": sorted(st["types"]),
                    "sample_dims": st["sample_dims"],
                }
                for pat, st in sorted(pattern_stats.items())
            },
            "layers": {
                str(blk): {
                    "total_bytes": layer_bytes[blk],
                    "exps_bytes": layer_exps_bytes[blk],
                }
                for blk in sorted(layer_bytes)
            },
        }
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved layout JSON: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
