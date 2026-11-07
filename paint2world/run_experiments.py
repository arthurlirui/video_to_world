"""Batch experiment runner: generates all measured data for the paper tables.

Produces:
  * results/main.json         — full method on 3 paintings
  * results/ablation.json     — 6 ablation variants on the landscape painting
  * results/trajectory.json   — 4 trajectory types on the landscape painting
  * results/scalability.json  — frame-count sweep {8,16,24,32} on landscape

Each entry is a metrics dict from paint2world.evaluation.run_evaluation.
Run:  python -m paint2world.run_experiments --out results
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import torch

from .evaluation import run_evaluation


def _cleanup():
    """Release GPU/CPU memory between runs to avoid OOM on long batches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _append_jsonl(path, record):
    """Append a single record to a JSON-Lines file (resilient to crashes)."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


PAINTINGS = [
    "data/sample_paintings/sample_00_landscape.png",
    "data/sample_paintings/sample_01_still_life.png",
    "data/sample_paintings/sample_02_portrait.png",
]
LANDSCAPE = PAINTINGS[0]


def _safe(p):
    if not os.path.exists(p):
        # Try absolute fallback.
        abs_p = os.path.join(os.path.dirname(__file__), "..", p)
        if os.path.exists(abs_p):
            return abs_p
    return p


def run_main(out_dir, num_frames=10, stage1_iters=40, stage3_iters=120):
    print("\n=== MAIN: full method on 3 paintings ===", flush=True)
    jsonl = os.path.join(out_dir, "main.jsonl")
    # Resume support: skip already-done scenes.
    done = set()
    if os.path.exists(jsonl):
        for line in open(jsonl):
            try:
                done.add(json.loads(line)["scene"])
            except Exception:
                pass
    results = []
    if os.path.exists(jsonl):
        for line in open(jsonl):
            try:
                results.append(json.loads(line))
            except Exception:
                pass
    for p in PAINTINGS:
        p = _safe(p)
        name = os.path.splitext(os.path.basename(p))[0]
        if name in done:
            print(f"  painting: {name} (cached, skipping)", flush=True)
            continue
        print(f"  painting: {name}", flush=True)
        m = run_evaluation(p, os.path.join(out_dir, "main", name), num_frames=num_frames,
                           stage1_iters=stage1_iters, stage3_iters=stage3_iters)
        m["scene"] = name
        results.append(m)
        _append_jsonl(jsonl, m)
        _cleanup()
    with open(os.path.join(out_dir, "main.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_ablation(out_dir, num_frames=10, stage1_iters=40, stage3_iters=120):
    print("\n=== ABLATION: 9 variants on landscape ===", flush=True)
    p = _safe(LANDSCAPE)
    jsonl = os.path.join(out_dir, "ablation.jsonl")
    variants = [
        ("full", {}),
        ("no_depth_warp", {"enable_depth_warp": False}),
        ("no_stroke_attn", {"enable_stroke_attn": False}),
        ("no_painting_anchor", {"enable_painting_anchor": False}),
        ("no_depth_warp_consistency", {"enable_depth_warp_consistency": False}),
        ("no_stroke_coherence", {"enable_stroke_coherence": False}),
        ("no_painting_view", {"enable_painting_view": False}),
        ("no_object_scene", {"enable_object_scene": False}),
        ("no_canvas_depth", {"enable_canvas_depth": False}),
        ("no_object_rigid", {"enable_object_rigid": False}),
        ("no_all_painting_controls", {
            "enable_depth_warp": False, "enable_stroke_attn": False,
            "enable_painting_anchor": False, "enable_depth_warp_consistency": False,
            "enable_stroke_coherence": False, "enable_painting_view": False,
        }),
    ]
    done = set()
    if os.path.exists(jsonl):
        for line in open(jsonl):
            try:
                done.add(json.loads(line)["variant"])
            except Exception:
                pass
    results = []
    if os.path.exists(jsonl):
        for line in open(jsonl):
            try:
                results.append(json.loads(line))
            except Exception:
                pass
    for vname, kwargs in variants:
        if vname in done:
            print(f"  variant: {vname} (cached, skipping)", flush=True)
            continue
        print(f"  variant: {vname}", flush=True)
        m = run_evaluation(p, os.path.join(out_dir, "ablation", vname), num_frames=num_frames,
                           stage1_iters=stage1_iters, stage3_iters=stage3_iters, **kwargs)
        m["variant"] = vname
        results.append(m)
        _append_jsonl(jsonl, m)
        _cleanup()
    with open(os.path.join(out_dir, "ablation.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_trajectory(out_dir, num_frames=10, stage1_iters=40, stage3_iters=120):
    print("\n=== TRAJECTORY: 4 types on landscape ===", flush=True)
    p = _safe(LANDSCAPE)
    jsonl = os.path.join(out_dir, "trajectory.jsonl")
    trajs = [
        ("orbit", {"trajectory": "orbit", "angle_deg": 25.0, "baseline": 0.18}),
        ("dolly", {"trajectory": "dolly", "baseline": 0.18}),
        ("pan", {"trajectory": "pan", "baseline": 0.18}),
        ("spiral", {"trajectory": "spiral", "angle_deg": 30.0, "baseline": 0.15}),
    ]
    done = set()
    if os.path.exists(jsonl):
        for line in open(jsonl):
            try:
                done.add(json.loads(line)["trajectory"])
            except Exception:
                pass
    results = []
    if os.path.exists(jsonl):
        for line in open(jsonl):
            try:
                results.append(json.loads(line))
            except Exception:
                pass
    for tname, kwargs in trajs:
        if tname in done:
            print(f"  trajectory: {tname} (cached, skipping)", flush=True)
            continue
        print(f"  trajectory: {tname}", flush=True)
        m = run_evaluation(p, os.path.join(out_dir, "trajectory", tname), num_frames=num_frames,
                           stage1_iters=stage1_iters, stage3_iters=stage3_iters, **kwargs)
        m["trajectory"] = tname
        results.append(m)
        _append_jsonl(jsonl, m)
        _cleanup()
    with open(os.path.join(out_dir, "trajectory.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_scalability(out_dir, stage1_iters=40, stage3_iters=120):
    print("\n=== SCALABILITY: frame-count sweep on landscape ===", flush=True)
    p = _safe(LANDSCAPE)
    jsonl = os.path.join(out_dir, "scalability.jsonl")
    frame_counts = [8, 12, 16, 20]
    done = set()
    if os.path.exists(jsonl):
        for line in open(jsonl):
            try:
                done.add(json.loads(line)["num_frames"])
            except Exception:
                pass
    results = []
    if os.path.exists(jsonl):
        for line in open(jsonl):
            try:
                results.append(json.loads(line))
            except Exception:
                pass
    for nf in frame_counts:
        if nf in done:
            print(f"  num_frames: {nf} (cached, skipping)", flush=True)
            continue
        print(f"  num_frames: {nf}", flush=True)
        m = run_evaluation(p, os.path.join(out_dir, "scalability", f"nf{nf}"), num_frames=nf,
                           stage1_iters=stage1_iters, stage3_iters=stage3_iters)
        m["num_frames"] = nf
        results.append(m)
        _append_jsonl(jsonl, m)
        _cleanup()
    with open(os.path.join(out_dir, "scalability.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--skip_scalability", action="store_true")
    parser.add_argument("--stage1_iters", type=int, default=40)
    parser.add_argument("--stage3_iters", type=int, default=120)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run_main(args.out, args.num_frames, stage1_iters=args.stage1_iters, stage3_iters=args.stage3_iters)
    run_ablation(args.out, args.num_frames, stage1_iters=args.stage1_iters, stage3_iters=args.stage3_iters)
    run_trajectory(args.out, args.num_frames, stage1_iters=args.stage1_iters, stage3_iters=args.stage3_iters)
    if not args.skip_scalability:
        run_scalability(args.out, stage1_iters=args.stage1_iters, stage3_iters=args.stage3_iters)
    print("\n=== ALL EXPERIMENTS DONE ===", flush=True)
    print(f"Results in {args.out}/")


if __name__ == "__main__":
    main()
