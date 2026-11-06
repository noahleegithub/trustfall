"""Check the SafeFall trip obstacle end to end, on a real (GPU) node.

What it does:
  1. asserts the obstacle body really exists in the built env (it is added by the terrain
     builder, so a silent miss here means every trip trial is a no-op), and prints its
     geom size / collision classes;
  2. prints the parked pose the model starts at;
  3. runs one pre-fall batch and prints, for a few trip trials, the sampled top-face
     height, the resolved direction and distance, and the world pose that was actually
     written into the backend -- read back from the model, plus the collision geom
     position when the backend exposes it;
  4. asserts the invariants (top face at the sampled height, unit direction, yaw matches,
     non-trip envs still parked).

Runs headless on a GPU node exactly like ``diag_parts.py`` (same env vars, no display)::

    CKPT=$SCRATCH/runs/<train>/.../model_025000.pt OUTDIR=$SCRATCH/diag/trip \\
      python scripts/safefall/diag_trip_obstacle.py

Optional env vars: ``TRIALS`` (default 16), ``TRIP_MODE`` (default ``obstacle``),
``PRESET`` (default ``paper``), ``DEVICE`` (default ``cuda``).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import benchmark as B

CKPT = os.environ["CKPT"]
OUTDIR = os.environ["OUTDIR"]
TRIALS = int(os.environ.get("TRIALS", "16"))


def obstacle_geom_ids(sim, body_id: int) -> list[int]:
    """MuJoCo geom ids owned by ``body_id``."""
    model = sim.root_model
    return [g for g in range(int(model.ngeom)) if int(model.geom_bodyid[g]) == body_id]


def readback(bench) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Per-world obstacle pose: positions ``[n, 3]``, quats ``wxyz`` ``[n, 4]``, geom xpos.

    Prefers the backend's per-world model arrays (that is what the collision pipeline
    reads); falls back to the single-env model for the classic backend.
    """
    body_id = bench.obstacle_body_id
    backend = bench.sim.backend
    geom_ids = obstacle_geom_ids(bench.sim, body_id)
    mjw_model = getattr(backend, "mjw_model", None)
    if mjw_model is not None:
        import warp as wp

        pos = wp.to_torch(mjw_model.body_pos).detach().cpu().numpy()[:, body_id]
        quat = wp.to_torch(mjw_model.body_quat).detach().cpu().numpy()[:, body_id]
        geom = None
        mjw_data = getattr(backend, "mjw_data", None)
        try:
            geom = wp.to_torch(mjw_data.geom_xpos).detach().cpu().numpy()[:, geom_ids]
        except Exception:  # the readback is diagnostic, not load-bearing
            geom = None
        return pos, quat, geom
    model = bench.sim.root_model
    return (
        model.body_pos[body_id][None].copy(),
        model.body_quat[body_id][None].copy(),
        model.geom_pos[geom_ids][None].copy() if geom_ids else None,
    )


def main() -> None:
    args = B.parse_args([
        "--checkpoint", CKPT,
        "--output-dir", OUTDIR,
        "--trials", str(TRIALS),
        "--num-envs", str(TRIALS),
        "--shard", "0",
        "--num-shards", "1",
        "--limit", str(TRIALS),
        "--preset", os.environ.get("PRESET", "paper"),
        "--trip-mode", os.environ.get("TRIP_MODE", "obstacle"),
        "--device", os.environ.get("DEVICE", "cuda"),
    ])
    bench = B.FallBenchmark(args)
    try:
        print("\n=== MODEL ===")
        print(f"trip mode: {args.trip_mode}")
        if args.trip_mode != "obstacle":
            print("not the obstacle mode; nothing to check")
            return
        body_id = bench.obstacle_body_id
        assert body_id is not None and body_id >= 0, "obstacle body missing from the compiled model"
        import mujoco

        name = mujoco.mj_id2name(bench.sim.root_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        geom_ids = obstacle_geom_ids(bench.sim, body_id)
        assert geom_ids, f"obstacle body {name} ({body_id}) has no geoms, so it cannot collide"
        m = bench.sim.root_model
        print(f"body '{name}' id={body_id} geoms={geom_ids}")
        for g in geom_ids:
            print(
                f"  geom '{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)}' "
                f"size={m.geom_size[g].tolist()} contype={int(m.geom_contype[g])} "
                f"conaffinity={int(m.geom_conaffinity[g])}"
            )
        print(f"half height={bench.obstacle_half_height} park z={bench.obstacle_park_z}")

        pos, quat, geom = readback(bench)
        print(f"\n=== POSE AT BUILD: z={np.round(pos[:, 2], 4).tolist()} ===")
        assert np.all(np.abs(pos[:, 2] - bench.obstacle_park_z) < 1e-5), "obstacle did not start parked"

        print(f"\n=== PRE-FALL BATCH ({TRIALS} trials) ===")
        scenarios = [bench.make_scenario(t) for t in range(TRIALS)]
        trip = [i for i, s in enumerate(scenarios) if "trip" in s.mechanisms]
        assert trip, "no trial in this batch has the trip mechanism; raise TRIALS"
        bench.run_prefall(scenarios)

        pos, quat, geom = readback(bench)
        print(f"{'trial':>5} {'height':>7} {'dir':>16} {'dist':>6} {'pose xyz':>22} {'yaw':>7} {'top z':>7}")
        for i in trip:
            s = scenarios[i]
            yaw = 2.0 * np.arctan2(quat[i, 2], quat[i, 3])
            print(
                f"{s.trial_id:5d} {s.trip_height:7.3f} "
                f"({s.trip_dir_x:6.3f},{s.trip_dir_y:6.3f}) {s.trip_distance:6.3f} "
                f"({pos[i, 0]:6.3f},{pos[i, 1]:6.3f},{pos[i, 2]:6.3f}) {yaw:7.3f} "
                f"{pos[i, 2] + bench.obstacle_half_height:7.3f}"
            )
            # Invariants of the placement rule.
            assert np.isfinite(pos[i]).all(), f"trial {s.trial_id}: non-finite obstacle pose"
            assert abs(pos[i, 2] + bench.obstacle_half_height - s.trip_height) < 1e-4, (
                f"trial {s.trial_id}: top face {pos[i, 2] + bench.obstacle_half_height:.4f} "
                f"!= sampled height {s.trip_height:.4f}"
            )
            assert abs(float(np.hypot(s.trip_dir_x, s.trip_dir_y)) - 1.0) < 1e-5, (
                f"trial {s.trial_id}: direction is not a unit vector"
            )
            assert abs(yaw - float(np.arctan2(s.trip_dir_y, s.trip_dir_x))) < 1e-4, (
                f"trial {s.trial_id}: box yaw {yaw:.4f} does not match travel heading"
            )
            if geom is not None:
                # The collision geom must have followed the body (forward kinematics alone
                # does not refresh a welded geom's collision pose).
                assert np.allclose(geom[i, 0, :2], pos[i, :2], atol=1e-4), (
                    f"trial {s.trial_id}: collision geom at {geom[i, 0]} != body at {pos[i]}"
                )

        nontrip = [i for i, s in enumerate(scenarios) if s.trip_height is None]
        if nontrip:
            z = pos[nontrip, 2]
            print(f"\nnon-trip envs ({len(nontrip)}) z: {np.round(z, 3).tolist()}")
            assert np.all(np.abs(z - bench.obstacle_park_z) < 1e-4), "a non-trip env's obstacle is not parked"
        print("\nall obstacle checks passed")
    finally:
        bench.close()


if __name__ == "__main__":
    main()
