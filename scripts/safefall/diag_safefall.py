"""Diagnostics for the SafeFall benchmark: baseline gain switching + illegal-contact positive control.

Part A  verify the protected-body mask and record the controller gains each baseline
        actually runs with, cross-checking recomputed metrics against the pilot CSV.
Part B  positive control: drop the robot upside down and confirm the illegal-contact
        detector fires with robot-appropriate protected patterns.

Run inside the SIF on a GPU node, from /workspace/holosoma.
"""
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import benchmark as B  # noqa: E402

CKPT = os.environ["CKPT"]
OUTDIR = os.environ["OUTDIR"]
RUN_DIR = os.environ.get("RUN_DIR", "")
TRIALS = [int(t) for t in os.environ.get("TRIALS", "4,9,23,30").split(",")]
EXTENDED = ("head", "hand", "wrist", "torso")


def build_bench():
    args = B.parse_args([
        "--checkpoint", CKPT,
        "--output-dir", OUTDIR,
        "--trials", "200",
        "--num-envs", "8",
        "--shard", "0",
        "--num-shards", "1",
        "--device", "cuda",
    ])
    return B.FallBenchmark(args)


def part_a(bench):
    print("\n=== PART A: protected mask + baseline gains ===")
    names = [str(n).lower() for n in bench.sim.body_names]
    mon = bench.monitor
    print(f"nbody={mon.nbody}  ndof={mon.ndof}")
    print(f"protected_patterns in use : {getattr(mon, 'protected_patterns', ('head','hand'))}")
    print(f"protected_mask.sum() (as configured) : {int(mon.protected_mask.sum())}")
    ext = mon._body_mask(names, EXTENDED)
    print(f"protected_mask.sum() (extended {EXTENDED}) : {int(ext.sum())}")
    print("  extended bodies:", [n for n, m in zip(names, ext.tolist()) if m])
    print(f"foot_mask.sum() : {int(mon.foot_mask.sum())}")
    print("  foot bodies:", [n for n, m in zip(names, mon.foot_mask.tolist()) if m])

    # record the gains each branch runs with
    orig_prepare = bench._prepare_branch
    seen = {"default": [], "damping": []}

    def wrapped(pre, baseline):
        ids = orig_prepare(pre, baseline)
        term = bench._joint_term()
        if baseline in seen and ids.numel():
            i = int(ids[0])
            seen[baseline].append((
                term._kp_scale[i].mean().item(),
                term._kd_scale[i].mean().item(),
                float(term.p_gains.mean()),
                float(term.d_gains.mean()),
            ))
        return ids

    bench._prepare_branch = wrapped

    for tid in TRIALS:
        sc = bench.make_scenario(tid)
        print(f"\n--- trial {tid}: {sc.mechanisms} ---")
        pre = bench.run_prefall([sc])
        if not bool(pre.valid[0]):
            print("  did not fall (valid=False); skipping branches")
            continue
        print(f"  t_step={int(pre.t_step[0])} trigger={int(pre.trigger_step[0])} "
              f"lead={int(pre.t_step[0]-pre.trigger_step[0])} steps")
        for b in ("default", "damping"):
            out = bench.run_branch(pre, b)
            kp, kd, p0, d0 = seen[b][-1]
            print(f"  {b:8s} kp_scale={kp:9.3e} kd_scale={kd:7.3f} "
                  f"(p_gain={p0:.2f} d_gain={d0:.2f}) "
                  f"contact={float(out['max_contact_force'][0]):9.1f}N "
                  f"illegal={float(out['illegal_contact'][0]):.0f}")

    # cross-check one nominal metric against the pilot CSV
    if RUN_DIR:
        path = os.path.join(RUN_DIR, "safefall", "trials_shard_0.csv")
        if os.path.exists(path):
            rows = {(int(r["trial_id"]), r["baseline"]): r for r in csv.DictReader(open(path))}
            print("\n=== PART A cross-check vs pilot CSV (nominal max_contact_force) ===")
            for tid in TRIALS:
                key = (tid, "nominal")
                if key not in rows:
                    continue
                csv_val = float(rows[key]["max_contact_force"])
                print(f"  trial {tid}: csv={csv_val:.1f} N")


def part_b(bench):
    print("\n=== PART B: illegal-contact positive control ===")
    mon = bench.monitor
    names = [str(n).lower() for n in bench.sim.body_names]
    mon.protected_mask = mon._body_mask(names, EXTENDED)
    print(f"protected bodies now: {int(mon.protected_mask.sum())}")

    n = bench.num_envs
    ids = torch.arange(n, device=bench.device)
    root = bench.sim.robot_root_states.clone()
    root[:, 0:3] = torch.tensor([0.0, 0.0, 1.6], device=bench.device)
    root[:, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=bench.device)  # 180 deg pitch
    root[:, 7:13] = 0.0
    bench.sim.set_actor_root_state_tensor_robots(ids, root[ids])

    dof = bench._dof_state_3d()
    dof[:] = 0.0
    bench.sim.set_dof_state_tensor_robots(ids, dof.reshape(-1, 2))

    mon.reset()
    steps = 40  # must be <= ImpactMonitor.window_steps (48)
    z_trace = []
    with torch.no_grad():
        for s in range(1, steps + 1):
            bench.env.step({"actions": bench._zeros_actions})
            z_trace.append(round(float(bench.sim.robot_root_states[:, 2].min()), 3))
        res = mon.results(ids, torch.ones_like(ids), torch.full_like(ids, steps))
    print(f"min root z over the drop: {min(z_trace)}  (trajectory every 10: {z_trace[::10]})")
    # which protected bodies saw ground contact at all?
    cf = bench.sim.contact_forces
    if hasattr(cf, "ndim") and cf.ndim == 3:
        per_body = cf.abs().amax(dim=(0, 2)).to(torch.float32)
        prot = [n for n, m in zip(names, mon.protected_mask.tolist()) if m]
        for n in prot:
            j = names.index(n)
            print(f"  peak contact force on {n}: {float(per_body[j]):.1f} N")
    ill = res.illegal_contact.to(torch.float32)
    print(f"upside-down drop: illegal_contact = {ill.tolist()}")
    print(f"                  max_contact_force = {[round(x,1) for x in res.max_contact_force.tolist()[:4]]}")
    print(f"                  max_limit_violations = {res.max_limit_violations.tolist()[:4]}")
    print("RESULT:", "POSITIVE CONTROL PASSED (detector fires)" if float(ill.max()) > 0
          else "POSITIVE CONTROL FAILED (no illegal contact detected)")


def main():
    bench = build_bench()
    try:
        part_a(bench)
        part_b(bench)
    finally:
        bench.close()


if __name__ == "__main__":
    main()
