"""Instrument the perturbation path and verify per-part geom attribution.

Prints:
  1. how many geoms exist and which geoms classify as head / torso / hand
  2. a short benchmark run (32 trials) with the diagnostic columns
  3. fallen vs recovered comparison of the diagnostics, to explain a low yield
"""
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import benchmark as B  # noqa: E402

CKPT = os.environ["CKPT"]
OUTDIR = os.environ["OUTDIR"]
TRIALS = int(os.environ.get("TRIALS", "32"))


def main() -> None:
    args = B.parse_args([
        "--checkpoint", CKPT,
        "--output-dir", OUTDIR,
        "--trials", str(TRIALS),
        "--num-envs", "64",
        "--shard", "0",
        "--num-shards", "1",
        "--limit", str(TRIALS),
        "--preset", os.environ.get("PRESET", "calibrated"),
        "--device", "cuda",
    ])
    bench = B.FallBenchmark(args)
    try:
        print("\n=== PART CLASSIFICATION ===")
        geoms = bench.sim.geom_names
        print(f"ngeom={len(geoms)}  (mj_id2name resolved {len(geoms)} names)")
        part = bench.monitor.geom_part.cpu()
        for pid, pname in enumerate(bench.monitor.PARTS):
            idx = (part == pid).nonzero(as_tuple=False).flatten().tolist()
            names = [geoms[i] for i in idx]
            print(f"  {pname:6s} {len(idx):2d} geoms: {names[:6]}{' ...' if len(names) > 6 else ''}")
        head_geoms = [geoms[i] for i in (part == 0).nonzero(as_tuple=False).flatten().tolist()]
        assert any("head" in g.lower() for g in head_geoms), f"no head geom found, got {head_geoms}"
        print("  -> head capsule IS attributed separately")

        print("\n=== RUN ===")
        bench.run()

        csv_path = os.path.join(OUTDIR, f"trials_shard_{args.shard}.csv")
        rows = list(csv.DictReader(open(csv_path)))
        print(f"\n=== RESULT: {len(rows)} rows ===")
        print("columns include perf diagnostics:",
              {"push_dv_norm", "root_z_at_pert", "root_z_min_post", "worst_hit_part"} <= set(rows[0]))

        import json
        scen = [json.loads(l) for l in open(os.path.join(OUTDIR, f"scenarios_shard_{args.shard}.jsonl"))]
        fell = {s["trial_id"] for s in scen if s.get("valid")}
        print(f"trials that fell: {len(fell)}/{len(scen)}")

        def stats(key, ids=None):
            vals = [float(r[key]) for r in rows if r["baseline"] == "nominal"]
            return (min(vals), sum(vals) / len(vals), max(vals)) if vals else (0, 0, 0)

        for key in ("push_dv_norm", "root_z_at_pert", "root_z_min_post"):
            lo, mean, hi = stats(key)
            fell_vals = [float(r[key]) for r in rows if r["baseline"] == "nominal" and int(r["trial_id"]) in fell]
            rec_vals = [float(r[key]) for r in rows if r["baseline"] == "nominal" and int(r["trial_id"]) not in fell]
            f_mean = sum(fell_vals) / len(fell_vals) if fell_vals else float("nan")
            r_mean = sum(rec_vals) / len(rec_vals) if rec_vals else float("nan")
            print(f"  {key:18s} min={lo:7.3f} mean={mean:7.3f} max={hi:7.3f} | fell_mean={f_mean:7.3f} recovered_mean={r_mean:7.3f}")

        print("\n=== PER-PART SEVERITY (nominal rows) ===")
        for key in ("peak_force_head", "peak_force_torso", "peak_force_hand",
                    "cum_impulse_head", "cum_impulse_torso", "cum_impulse_hand"):
            vals = [float(r[key]) for r in rows if r["baseline"] == "nominal"]
            if vals:
                print(f"  {key:20s} mean={sum(vals)/len(vals):9.1f} max={max(vals):9.1f}")
        from collections import Counter
        print("  worst_hit_part counts:", dict(Counter(r["worst_hit_part"] for r in rows)))
    finally:
        bench.close()


if __name__ == "__main__":
    main()
