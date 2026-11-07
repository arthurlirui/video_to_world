"""Convert measured experiment JSON results into LaTeX table rows for the paper.

Reads results/{main,ablation,trajectory,scalability}.json and prints
ready-to-paste LaTeX rows. Run after run_experiments.py completes.
"""

from __future__ import annotations

import json
import os
import sys


def _fmt(x, digits=3):
    if x is None:
        return "--"
    return f"{x:.{digits}f}"


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results"

    # ---- Derived metric helpers ----
    # We map raw measured metrics to paper-facing columns:
    #   Align. Residual  = align_loss_final (lower = better canonical geometry)
    #   Temporal Smooth. = 100*(1 - temporal_smoothness) (higher = more stable video)
    #   Painting Fid.    = painting_lpips (lower = better)
    #   Painting L1      = painting_l1 (lower = better)
    #   Stroke Coh.      = stroke_coherence (lower = more coherent strokes)
    def align_resid(d):
        return d.get("align_loss_final", 0.0)

    def consistency(d):
        return 100.0 * d.get("align_loss_drop", 0.0)

    def photo(d):
        return 100.0 * (1.0 - d.get("temporal_smoothness", 0.0))

    def paint_fid(d):
        return d.get("painting_lpips", 0.0)

    def stroke_coh(d):
        return d.get("stroke_coherence", 0.0)

    def l1(d):
        return d.get("painting_l1", 0.0)

    def rt(d):
        return d.get("runtime_s", 0.0)

    def n_obj(d):
        return d.get("num_objects", 0) or 0

    def n_rel(d):
        return d.get("num_relations", 0) or 0

    def canvas_l1(d):
        return d.get("canvas_painting_l1")

    def obj_depth(d):
        return d.get("object_depth_consistency")

    def rel_drift(d):
        return d.get("relation_gap_drift")

    # ---- Table 1: Main (per scene + average) ----
    print("\n=== TABLE: Main comparison (per scene) ===")
    main = _load(os.path.join(root, "main.json"))
    if main:
        print(f"{'Scene':<28} {'3DConsist':>9} {'PhotoCons':>10} {'PaintL1':>8} {'PaintFid':>8} {'StrokeCoh':>9} {'RT(s)':>7}")
        acc = {k: 0.0 for k in ["c", "p", "l1", "pf", "sc", "rt"]}
        for r in main:
            c, p, l1v, pf, sc, t = consistency(r), photo(r), l1(r), paint_fid(r), stroke_coh(r), rt(r)
            acc["c"] += c; acc["p"] += p; acc["l1"] += l1v; acc["pf"] += pf; acc["sc"] += sc; acc["rt"] += t
            print(f"{r['scene']:<28} {c:>9.2f} {p:>10.2f} {_fmt(l1v):>8} {_fmt(pf):>8} {_fmt(sc,4):>9} {t:>7.1f}")
        n = len(main)
        print(f"{'Average':<28} {acc['c']/n:>9.2f} {acc['p']/n:>10.2f} {_fmt(acc['l1']/n):>8} {_fmt(acc['pf']/n):>8} {_fmt(acc['sc']/n,4):>9} {acc['rt']/n:>7.1f}")

    # ---- Table 2: Ablation ----
    print("\n=== TABLE: Ablation ===")
    abl = _load(os.path.join(root, "ablation.json"))
    if abl:
        print(f"{'Variant':<30} {'3DConsist':>9} {'PhotoCons':>10} {'PaintL1':>8} {'PaintFid':>8} {'StrokeCoh':>9} {'RT(s)':>7}")
        for r in abl:
            c, p, l1v, pf, sc, t = consistency(r), photo(r), l1(r), paint_fid(r), stroke_coh(r), rt(r)
            print(f"{r['variant']:<30} {c:>9.2f} {p:>10.2f} {_fmt(l1v):>8} {_fmt(pf):>8} {_fmt(sc,4):>9} {t:>7.1f}")

    # ---- Table 3: Trajectory comparison ----
    print("\n=== TABLE: Trajectory comparison ===")
    traj = _load(os.path.join(root, "trajectory.json"))
    if traj:
        print(f"{'Trajectory':<14} {'3DConsist':>9} {'PhotoCons':>10} {'PaintL1':>8} {'PaintFid':>8} {'StrokeCoh':>9} {'RT(s)':>7}")
        for r in traj:
            c, p, l1v, pf, sc, t = consistency(r), photo(r), l1(r), paint_fid(r), stroke_coh(r), rt(r)
            print(f"{r['trajectory']:<14} {c:>9.2f} {p:>10.2f} {_fmt(l1v):>8} {_fmt(pf):>8} {_fmt(sc,4):>9} {t:>7.1f}")

    # ---- Table 4: Scalability ----
    print("\n=== TABLE: Scalability ===")
    sc = _load(os.path.join(root, "scalability.json"))
    if sc:
        print(f"{'#Frames':>8} {'3DConsist':>9} {'PhotoCons':>10} {'PaintL1':>8} {'PaintFid':>8} {'#Gauss':>7} {'RT(s)':>7}")
        for r in sc:
            c, p, l1v, pf, t = consistency(r), photo(r), l1(r), paint_fid(r), rt(r)
            ng = r.get("num_gaussians", 0)
            print(f"{r['num_frames']:>8} {c:>9.2f} {p:>10.2f} {_fmt(l1v):>8} {_fmt(pf):>8} {ng:>7} {t:>7.1f}")

    # ---- Table 5: Object-level scene construction ----
    print("\n=== TABLE: Object scene (main runs) ===")
    if main:
        print(f"{'Scene':<28} {'#Obj':>5} {'#Rel':>5} {'#Adj':>5} {'CanvasL1':>9} {'ObjDepth':>9} {'RelDrift':>9}")
        for r in main:
            adj = r.get("placement_adjusted", 0) or 0
            print(f"{r['scene']:<28} {n_obj(r):>5} {n_rel(r):>5} {adj:>5} "
                  f"{_fmt(canvas_l1(r)):>9} {_fmt(obj_depth(r)):>9} {_fmt(rel_drift(r), 5):>9}")

    print("\n=== TABLE: Object-branch ablation ===")
    if abl:
        print(f"{'Variant':<30} {'#Obj':>5} {'#Rel':>5} {'PaintL1':>8} {'ObjDepth':>9} {'RelDrift':>9}")
        for r in abl:
            print(f"{r['variant']:<30} {n_obj(r):>5} {n_rel(r):>5} {_fmt(l1(r)):>8} "
                  f"{_fmt(obj_depth(r)):>9} {_fmt(rel_drift(r), 5):>9}")

    # ---- LaTeX row output ----
    print("\n=== LATEX ROWS (main, averaged) ===")
    if main:
        n = len(main)
        avg_c = sum(consistency(r) for r in main) / n
        avg_p = sum(photo(r) for r in main) / n
        avg_l1 = sum(l1(r) for r in main) / n
        avg_pf = sum(paint_fid(r) for r in main) / n
        avg_sc = sum(stroke_coh(r) for r in main) / n
        avg_rt = sum(rt(r) for r in main) / n
        print(f"Paint2World (ours) & {avg_c:.2f} & {avg_p:.2f} & {avg_l1:.3f} & {avg_pf:.3f} & {avg_sc:.4f} & {avg_rt:.1f}")

    print("\n=== LATEX ROWS (ablation) ===")
    if abl:
        for r in abl:
            v = r["variant"].replace("_", "\\_")
            c = consistency(r); p = photo(r); l1v = l1(r); pf = paint_fid(r); s = stroke_coh(r); t = rt(r)
            print(f"{v} & {c:.2f} & {p:.2f} & {l1v:.3f} & {pf:.3f} & {s:.4f} & {t:.1f} \\\\")

    print("\n=== LATEX ROWS (trajectory) ===")
    if traj:
        for r in traj:
            tr = r["trajectory"]
            c = consistency(r); p = photo(r); l1v = l1(r); pf = paint_fid(r); s = stroke_coh(r); t = rt(r)
            print(f"{tr} & {c:.2f} & {p:.2f} & {l1v:.3f} & {pf:.3f} & {s:.4f} & {t:.1f} \\\\")

    print("\n=== LATEX ROWS (scalability) ===")
    if sc:
        for r in sc:
            nf = r["num_frames"]
            c = consistency(r); p = photo(r); l1v = l1(r); pf = paint_fid(r); ng = r.get("num_gaussians",0); t = rt(r)
            print(f"{nf} & {c:.2f} & {p:.2f} & {l1v:.3f} & {pf:.3f} & {ng} & {t:.1f} \\\\")

    print("\n=== LATEX ROWS (object scene, main) ===")
    if main:
        for r in main:
            sn = r["scene"].replace("_", "\\_")
            adj = r.get("placement_adjusted", 0) or 0
            print(f"{sn} & {n_obj(r)} & {n_rel(r)} & {adj} & "
                  f"{_fmt(canvas_l1(r))} & {_fmt(obj_depth(r))} & {_fmt(rel_drift(r), 5)} \\\\")

    print("\n=== LATEX ROWS (ablation, object columns) ===")
    if abl:
        for r in abl:
            v = r["variant"].replace("_", "\\_")
            print(f"{v} & {n_obj(r)} & {n_rel(r)} & {_fmt(l1(r))} & "
                  f"{_fmt(obj_depth(r))} & {_fmt(rel_drift(r), 5)} \\\\")


if __name__ == "__main__":
    main()
