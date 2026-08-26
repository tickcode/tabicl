#!/usr/bin/env python3
"""Export a TabICL checkpoint (.ckpt with {"config", "state_dict"}) to GGUF
for the C++ inference library under cpp/.

Usage:
    python scripts/export_gguf.py <input.ckpt> <output.gguf> [--manifest out.json]

The exporter validates that the checkpoint's config is within the feature set
implemented by the C++ port and fails loudly otherwise.  All tensors are
written as fp32 with deterministically shortened names (ggml caps tensor names
at 64 bytes); the exact mapping is embedded as `tabicl.tensor_name_map` and a
SHA256-per-tensor manifest is written alongside for the loader parity test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

# Ordered prefix/substring replacements. Shared contract with
# cpp/src/gguf_model.cpp (which stores the *shortened* names).
NAME_REPLACEMENTS = [
    ("col_embedder.tf_col.blocks.", "col.blk."),
    ("col_embedder.", "col."),
    ("row_interactor.tf_row.blocks.", "row.blk."),
    ("row_interactor.tf_row.", "row."),
    ("row_interactor.", "row."),
    ("icl_predictor.tf_icl.blocks.", "icl.blk."),
    ("icl_predictor.", "icl."),
    ("multihead_attn1.attn.", "attn1."),
    ("multihead_attn2.attn.", "attn2."),
    ("multihead_attn1.", "attn1."),
    ("multihead_attn2.", "attn2."),
    ("ssmax_layer.", "ssmax."),
]

GGML_MAX_NAME = 64

# Feature matrix the C++ port implements. Anything outside this set must fail
# here, not silently produce wrong numbers at inference time.
SUPPORTED = {
    "col_affine": [False],
    "col_feature_group": ["same"],
    "col_target_aware": [True],
    "col_ssmax": ["qassmax-mlp-elementwise"],
    "icl_ssmax": ["qassmax-mlp-elementwise"],
    "activation": ["gelu"],
    "norm_first": [True],
    "dropout": [0.0],
    "row_rope_interleaved": [False],  # v2 checkpoints; v1 (interleaved) not ported
}


def shorten(name: str) -> str:
    for old, new in NAME_REPLACEMENTS:
        name = name.replace(old, new)
    return name


def validate_config(config: dict) -> None:
    errors = []
    for key, allowed in SUPPORTED.items():
        if key in config and config[key] not in allowed:
            errors.append(f"  {key} = {config[key]!r} (supported: {allowed})")
    if errors:
        raise SystemExit(
            "checkpoint config is outside the C++ port's supported feature set:\n"
            + "\n".join(errors)
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="TabICL .ckpt path")
    ap.add_argument("output", type=Path, help="output .gguf path")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="SHA256 manifest path (default: <output>.manifest.json)")
    args = ap.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=True)
    if set(ckpt.keys()) != {"config", "state_dict"}:
        raise SystemExit(f"unexpected checkpoint keys: {sorted(ckpt.keys())}")
    config, state_dict = ckpt["config"], ckpt["state_dict"]
    validate_config(config)

    task = "regression" if config["max_classes"] == 0 else "classification"

    name_map = {}
    for key in state_dict:
        short = shorten(key)
        if len(short) >= GGML_MAX_NAME:
            raise SystemExit(f"shortened name still too long ({len(short)}): {short}")
        if short in name_map.values():
            raise SystemExit(f"name collision after shortening: {short}")
        name_map[key] = short

    writer = GGUFWriter(str(args.output), arch="tabicl")
    writer.add_string("tabicl.task", task)
    writer.add_uint32("tabicl.format_version", 1)
    writer.add_uint32("tabicl.max_classes", config["max_classes"])
    writer.add_uint32("tabicl.num_quantiles", config["num_quantiles"])
    writer.add_uint32("tabicl.embed_dim", config["embed_dim"])
    writer.add_uint32("tabicl.col.num_blocks", config["col_num_blocks"])
    writer.add_uint32("tabicl.col.num_heads", config["col_nhead"])
    writer.add_uint32("tabicl.col.num_inds", config["col_num_inds"])
    writer.add_uint32("tabicl.col.feature_group_size", config["col_feature_group_size"])
    writer.add_uint32("tabicl.row.num_blocks", config["row_num_blocks"])
    writer.add_uint32("tabicl.row.num_heads", config["row_nhead"])
    writer.add_uint32("tabicl.row.num_cls", config["row_num_cls"])
    writer.add_float32("tabicl.row.rope_base", float(config["row_rope_base"]))
    writer.add_uint32("tabicl.icl.num_blocks", config["icl_num_blocks"])
    writer.add_uint32("tabicl.icl.num_heads", config["icl_nhead"])
    writer.add_uint32("tabicl.ff_factor", config["ff_factor"])
    writer.add_bool("tabicl.bias_free_ln", config.get("bias_free_ln", False))
    writer.add_float32("tabicl.norm_eps", 1e-5)
    writer.add_float32("tabicl.softmax_temperature", 0.9)
    writer.add_string("tabicl.config_json", json.dumps(config, sort_keys=True))
    writer.add_string("tabicl.tensor_name_map", json.dumps(name_map, sort_keys=True))

    manifest = {"source": str(args.input), "task": task, "tensors": {}}
    for key, tensor in state_dict.items():
        arr = tensor.detach().cpu().contiguous().numpy()
        if arr.dtype != np.float32:
            raise SystemExit(f"non-fp32 tensor {key}: {arr.dtype}")
        short = name_map[key]
        writer.add_tensor(short, arr)
        manifest["tensors"][short] = {
            "orig_name": key,
            "shape": list(arr.shape),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"wrote {args.output} ({len(state_dict)} tensors, task={task})")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
