#!/usr/bin/env python3
"""
Build a hybrid GPTQ-INT4 + FP8 checkpoint for Qwen3.5-122B-A10B.

Takes MoE expert weights from the GPTQ-INT4 checkpoint (0.5 bytes/param),
and dense layers (attention, shared experts, embeddings) from the official
FP8 checkpoint (1 byte/param + calibrated block scales).

Result: a checkpoint that is ~9 GB smaller than the GPTQ-INT4 original
while using properly calibrated FP8 scales (not naive cast), yielding
better decode throughput on bandwidth-limited hardware.

NOTE: Requires the hybrid FP8 dispatch patch from https://github.com/rmstxrx/vllm/tree/v0.17.1-hybrid-fp8

Usage:
    python build-hybrid-checkpoint.py \
        --gptq-dir  ~/inference/models/hf/qwen3.5-122b-a10b-gptq-int4 \
        --fp8-repo  Qwen/Qwen3.5-122B-A10B-FP8 \
        --output    ~/inference/models/hf/qwen3.5-122b-a10b-fp8hybrid
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def get_fp8_non_expert_manifest(fp8_repo: str) -> dict[str, str]:
    """Get the weight_map for non-expert tensors from FP8 checkpoint."""
    idx_path = hf_hub_download(fp8_repo, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    
    wm = idx["weight_map"]
    return {k: v for k, v in wm.items() if ".experts." not in k}


def download_fp8_shards(fp8_repo: str, shards: set[str], cache_dir: Path) -> dict[str, Path]:
    """Download only the needed shards from the FP8 checkpoint."""
    shard_paths = {}
    for shard in sorted(shards):
        print(f"  Downloading {shard}...", flush=True)
        path = hf_hub_download(fp8_repo, shard, local_dir=str(cache_dir))
        shard_paths[shard] = Path(path)
        print(f"    -> {path}")
    return shard_paths


def extract_fp8_tensors(shard_paths: dict[str, Path], wanted: dict[str, str]) -> dict[str, torch.Tensor]:
    """Extract non-expert tensors from downloaded FP8 shards."""
    tensors = {}
    for shard_name, shard_path in shard_paths.items():
        keys_in_shard = [k for k, v in wanted.items() if v == shard_name]
        if not keys_in_shard:
            continue
        
        print(f"  Extracting {len(keys_in_shard)} tensors from {shard_name}...")
        with safe_open(str(shard_path), framework="pt") as f:
            for k in keys_in_shard:
                tensors[k] = f.get_tensor(k)
    
    return tensors


def build_hybrid_checkpoint(
    gptq_dir: Path,
    fp8_tensors: dict[str, torch.Tensor],
    output_dir: Path,
):
    """Build hybrid checkpoint: GPTQ experts + FP8 dense layers."""
    
    # 1. Copy non-safetensors files
    for f in gptq_dir.iterdir():
        if f.suffix != ".safetensors" and not f.name.startswith("."):
            if f.is_file():
                shutil.copy2(f, output_dir / f.name)
    
    # 2. Process each GPTQ shard
    gptq_shards = sorted(gptq_dir.glob("model.safetensors-*"))
    total_replaced = 0
    total_added = 0
    total_saved_bytes = 0
    
    # Track which FP8 tensors have been placed
    placed_fp8 = set()
    
    for i, shard_path in enumerate(gptq_shards):
        shard_name = shard_path.name
        print(f"  [{i+1}/{len(gptq_shards)}] {shard_name}...", end=" ", flush=True)
        
        gptq_tensors = load_file(str(shard_path))
        output_tensors = {}
        replaced = 0
        
        for name, tensor in gptq_tensors.items():
            if name in fp8_tensors:
                # Replace BF16 tensor with FP8 version
                fp8_tensor = fp8_tensors[name]
                old_bytes = tensor.numel() * tensor.element_size()
                new_bytes = fp8_tensor.numel() * fp8_tensor.element_size()
                total_saved_bytes += old_bytes - new_bytes
                output_tensors[name] = fp8_tensor
                placed_fp8.add(name)
                replaced += 1
                
                # Also add the scale tensor if it exists
                scale_name = name.replace(".weight", ".weight_scale_inv")
                if scale_name in fp8_tensors and scale_name != name:
                    output_tensors[scale_name] = fp8_tensors[scale_name]
                    placed_fp8.add(scale_name)
                    total_added += 1
            else:
                output_tensors[name] = tensor
        
        total_replaced += replaced
        
        save_file(output_tensors, str(output_dir / shard_name))
        print(f"replaced={replaced}, tensors={len(output_tensors)}")
    
    # 3. Check for any FP8 tensors not yet placed (e.g. scale tensors
    #    for weights that exist in shards we already processed)
    unplaced = set(fp8_tensors.keys()) - placed_fp8
    if unplaced:
        print(f"\n  Adding {len(unplaced)} unplaced FP8 tensors to final shard...")
        # Load last shard, add them, re-save
        last_shard = output_dir / gptq_shards[-1].name
        existing = load_file(str(last_shard))
        for name in unplaced:
            existing[name] = fp8_tensors[name]
            total_added += 1
        save_file(existing, str(last_shard))
    
    return total_replaced, total_added, total_saved_bytes


def update_safetensors_index(output_dir: Path):
    """Rebuild model.safetensors.index.json from actual shard contents."""
    weight_map = {}
    total_size = 0
    
    for shard_path in sorted(output_dir.glob("model.safetensors-*")):
        with safe_open(str(shard_path), framework="pt") as f:
            for key in f.keys():
                weight_map[key] = shard_path.name
                tensor = f.get_tensor(key)
                total_size += tensor.numel() * tensor.element_size()
    
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map
    }
    
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    
    print(f"  Index rebuilt: {len(weight_map)} tensors, {total_size/1e9:.2f} GB")


def update_config(output_dir: Path):
    """Update config.json with hybrid quantization metadata."""
    config_path = output_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    config["_hybrid_quant_info"] = {
        "description": "Hybrid GPTQ-INT4 + FP8 checkpoint for single-GPU deployment",
        "moe_experts": "GPTQ INT4 (group_size=128, sym=True, Marlin kernels)",
        "dense_layers": "FP8 E4M3 block-128 (from official Qwen/Qwen3.5-122B-A10B-FP8, calibrated scales)",
        "norms_gates_embeddings": "Preserved at source dtype (BF16 for norms/gates, FP8 for others)",
        "source_gptq": "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
        "source_fp8": "Qwen/Qwen3.5-122B-A10B-FP8",
        "vllm_patch": "https://github.com/rmstxrx/vllm/tree/v0.17.1-hybrid-fp8",
        "target_hardware": "NVIDIA DGX Spark (GB10, 128GB unified, 273 GB/s)",
        "converter": "build-hybrid-checkpoint.py"
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build hybrid GPTQ-INT4 + FP8 checkpoint")
    parser.add_argument("--gptq-dir", required=True, help="Path to GPTQ-INT4 model")
    parser.add_argument("--fp8-repo", default="Qwen/Qwen3.5-122B-A10B-FP8", help="HF repo for FP8 model")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    gptq_dir = Path(args.gptq_dir)
    output_dir = Path(args.output)
    
    print("=== Hybrid GPTQ-INT4 + FP8 Checkpoint Builder ===")
    print(f"  GPTQ source: {gptq_dir}")
    print(f"  FP8 source:  {args.fp8_repo}")
    print(f"  Output:      {output_dir}")
    print()
    
    # Step 1: Get manifest of non-expert tensors from FP8 checkpoint
    print("[1/5] Fetching FP8 tensor manifest...")
    fp8_manifest = get_fp8_non_expert_manifest(args.fp8_repo)
    shards_needed = set(fp8_manifest.values())
    print(f"  Non-expert tensors: {len(fp8_manifest)}")
    print(f"  Shards to download: {shards_needed}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would download shards and build hybrid. Exiting.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".fp8_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Step 2: Download needed FP8 shards
    print(f"\n[2/5] Downloading {len(shards_needed)} FP8 shards...")
    shard_paths = download_fp8_shards(args.fp8_repo, shards_needed, cache_dir)
    
    # Step 3: Extract non-expert FP8 tensors
    print(f"\n[3/5] Extracting FP8 tensors...")
    fp8_tensors = extract_fp8_tensors(shard_paths, fp8_manifest)
    print(f"  Extracted: {len(fp8_tensors)} tensors")
    
    # Show dtype breakdown
    dtypes = {}
    for name, t in fp8_tensors.items():
        d = str(t.dtype)
        dtypes[d] = dtypes.get(d, 0) + 1
    print(f"  Dtypes: {dtypes}")
    
    # Step 4: Build hybrid checkpoint
    print(f"\n[4/5] Building hybrid checkpoint...")
    replaced, added, saved = build_hybrid_checkpoint(gptq_dir, fp8_tensors, output_dir)
    
    # Step 5: Update index and config
    print(f"\n[5/5] Updating index and config...")
    update_safetensors_index(output_dir)
    update_config(output_dir)
    
    # Cleanup downloaded FP8 shards
    shutil.rmtree(cache_dir, ignore_errors=True)
    
    print(f"\n=== Complete ===")
    print(f"  Tensors replaced (BF16→FP8):  {replaced}")
    print(f"  Scale tensors added:           {added}")
    print(f"  Bytes saved:                   {saved/1e9:.2f} GB")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    exit(main() or 0)
