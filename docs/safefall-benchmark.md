# SafeFall baseline benchmark

This is the measurement + baseline-evaluation half of a SafeFall replication
(Meng et al., [arXiv:2511.18509](https://arxiv.org/abs/2511.18509)). It does **not**
train the fall predictor or the protective policy yet. It runs walking + perturbation
fall trials on a trained nominal locomotion policy and evaluates the paper's three
baselines from the same trigger state:

| baseline | controller after `T - 300 ms` |
|---|---|
| `nominal` | the locomotion policy continues through the fall |
| `default` | hold the default joint targets with nominal PD gains |
| `damping` | `Kp ≈ 0`, `Kd = 10`, constant target, zero target velocity |

## Metrics

Implemented in `holosoma/simulator/mujoco/impact_metrics.py` and sampled at every
physics substep:

| metric | signal | definition |
|---|---|---|
| `max_joint_torque` | `qfrc_constraint` | peak `|τ|` over the 29 robot joints (constraint torque, excluding motor output) |
| `max_torque_ratio` | `qfrc_constraint` | peak `|τ| / τ_max` using `dof_effort_limit_list` |
| `max_joint_reaction` | `cfrc_int` | peak force transmitted through each joint's child body |
| `max_contact_force` | per-contact solver force | peak per-link **ground** contact force (non-robot body only) |
| `max_impulse` | per-contact force | max per-physics-step ground reaction impulse, `|ΣF| dt` |
| `cumulative_impulse` | per-contact force | sum of per-step impulses over the trial window |
| `illegal_contact` | per-contact body pair | 1 if a protected link (head / hand — `wrist` on a handless humanoid) contacted ground or a non-adjacent link |
| `torso_contact` | per-contact body pair | 1 if the torso struck the ground. Reported separately: every fall lands on the torso, so folding it into `illegal_contact` saturates that metric at 100% |
| `max_limit_violations` | `qpos` vs hard limits | max number of joints outside position limits at one substep |

`cfrc_int` and `cfrc_ext` are `[torque(3), force(3)]`; the code uses the force half.

## Perturbation presets

`--preset` selects the perturbation strength. The active preset is recorded in
`manifest.json`; quote it with any result.

| parameter | `paper` | `calibrated` |
|---|---|---|
| push `|dv|` | 1.5–3.5 m/s | 2.5–4.5 m/s |
| slip friction | 0.02–0.15 | 0.01–0.08 |
| slip duration | 0.10–0.30 s | 0.20–0.50 s |
| trip (default, obstacle mode): top-face height | 0.05–0.15 m | 0.05–0.15 m |
| trip (force mode only): force | 150–400 N | 400–800 N |
| trip (force mode only): duration | 0.05 s | 0.10 s |
| freeze motor-off probability | 0.5 | 0.8 |
| freeze two-joint probability | 0.30 | 0.55 |

Measured fall yields on the G1 (200 trials, 64 environments, seed 0):

Trip mode changes the verdict on that factor by more than an order of magnitude
(200 trials each, paper preset, same seed):

| trip mode | trip trials that fell | overall yield | head peak (max) | torso peak (max) |
|---|---|---|---|---|
| `obstacle` (5–15 cm box) | **96%** | **50%** | 20.4 kN | 19.0 kN |
| `force` (150–400 N) | 29% | 25% | 10.2 kN | 13.9 kN |

Non-trip trials are unaffected by mode (20% vs 22% fell), so the difference is entirely
the hazard: a real foot-catch pitches the robot forward onto its head, whereas the
synthetic foot force tends to topple it sideways onto the torso. The 5–15 cm band is
kept deliberately — a near-certain trip is the honest model of a swing foot meeting an
obstacle, and the paper specifies no target yield. Treat 50% (obstacle) as the current
overall figure; the 26% figure below is the earlier force-mode measurement.

| preset | yield | freeze | push | slip | trip |
|---|---|---|---|---|---|
| `paper` | **26%** | 28% | 40% | 14% | 22% |
| `calibrated` | **62%** | 58% | 74% | 22% | 92% |

`paper` is the default: it reproduces the write-up's perturbation regime and gives
~1,300 usable falls from a 5,000-trial run, which is ample. `calibrated` is a
separate stress profile, not a correction of `paper`.

These per-mechanism figures were measured in **force** mode (synthetic trip). With the
default obstacle trip the trip row becomes 96%, and overall yield ~50%.

An earlier version of this note reported a 3-5% yield at `paper` strength and argued
the ranges were too weak. That measurement was invalid — a spawn bug was placing
trials off the ground plane, so most robots free-fell without ever touching ground
(see the note on spawn position below). Do not tune perturbation strength against a
yield that has not been checked against the terrain extent.

## Fall scenarios

A trial is: walk 1-2 s with the nominal policy, then apply 1-3 of:

- **push** — horizontal base velocity perturbation, 1.5-3.5 m/s
- **slip** — lower the foot sliding friction to `[0.02, 0.15]` for 0.1-0.3 s
- **trip** — a box hazard placed in front of the robot at the trigger (`--trip-mode
  obstacle`, default); `--trip-mode force` keeps the original 150-400 N horizontal force
  on the swing foot for ~50 ms
- **freeze** — motor-off (zero torque) or brake-lock (hold the failure angle) on 1-2 joints

The first non-foot ground contact after the perturbation is `T`. The oracle trigger is
`T - 300 ms`; the evaluated window is 40 control steps (0.8 s at 50 Hz). The nominal
baseline runs continuously; `default`/`damping` branch from the saved trigger state.

The benchmark disables domain randomization and termination so all baselines are paired
and the post-impact sequence is not cut short.

### Trip hazard: box obstacle vs synthetic force

`--trip-mode obstacle` (default) is a real hazard: a welded 30 x 30 cm box, 10 cm tall,
added to the scene by the terrain builder (`TerrainTermCfg.obstacle`) and moved every
trial at the moment the trip factor fires, so the swing foot physically catches on it.

Placement rule, applied per trial at the trigger (implementation:
`scripts/safefall/trip_placement.py`, unit-tested in
`src/holosoma/tests/test_trip_placement.py`):

| quantity | rule |
|---|---|
| direction | unit base velocity (pelvis xy, `|v| >= 0.15 m/s`), else the commanded heading, else `+x` |
| distance | `clamp(speed * 0.30 s, 0.25 m, 1.0 m)` — `--trip-lead-time`, `--trip-min-distance`, `--trip-max-distance` |
| top-face height | sampled per trial from `[0.05, 0.15]` m — `--trip-height-min/--trip-height-max` |
| box centre z | `trip_height - 0.05`, so the top face sits at the sampled height and the lower half sinks into the ground |
| yaw | `atan2(dir_y, dir_x)`; the box face is perpendicular to travel |

Outside the trip window the box is parked at `z = -5 m` (below the floor), so non-trip
trials and batches without a trip see an unmodified scene. It stays where it was placed
through `reset_all()` — the branch resets restore robot state and friction, not model
geometry — so the `default` and `damping` branches face the same hazard as `nominal`.

The horizontal speed also sets the distance: a robot walking at 1 m/s gets the box 0.30 m
ahead, a standing robot 0.25 m. Because the box is placed at the trigger and not scripted
afterwards, the trip is only as repeatable as the walking state at that instant; the
resolved height/direction/distance are written to each CSV row (`trip_mode`,
`trip_height`, `trip_distance`, `trip_dir_x`, `trip_dir_y`) and to the scenario JSONL, and
`scripts/safefall/diag_trip_obstacle.py` prints the world pose actually applied.

`--trip-mode force` keeps the old behaviour byte-for-byte (150-400 N on the swing foot
for ~50 ms at `paper` strength) for reproducing earlier runs. The trip force columns are
populated in that mode only; in obstacle mode `trip_force` is `0.0` and
`trip_height`/`trip_distance`/`trip_dir_*` carry the hazard. Both modes write the same
columns, so a mixed-mode output directory still passes the header check — check
`trip_mode` (it is in every row and in `manifest.json`) before pooling two runs.

## HPRC / ACES workflow

```bash
# 1. Pilot (one shard, 200 trials)
CHECKPOINT=$SCRATCH/runs/<train>/.../model_025000.pt \
  TRIALS=200 SHARDS=1 NUM_ENVS=64 SEED=0 \
  sbatch --array=0 scripts/safefall/bench.sbatch

# 2. Full run (8 shards, 5000 trials)
CHECKPOINT=$SCRATCH/runs/<train>/.../model_025000.pt \
  TRIALS=5000 SHARDS=8 NUM_ENVS=256 SEED=0 \
  sbatch --array=0-7 scripts/safefall/bench.sbatch

# 3. Report (login node or CPU node)
RUN_ROOT=$SCRATCH/runs/<array_job_id> bash scripts/safefall/report.sh

# 4. Optional videos for selected trials
CHECKPOINT=$SCRATCH/runs/<train>/.../model_025000.pt \
  TRIAL_IDS="3 17 42" BASELINES="nominal default damping" \
  RUN_ROOT=$SCRATCH/runs/<render_job_id> \
  sbatch scripts/safefall/render.sbatch
```

Outputs (under `$SCRATCH/runs/<array_job_id>/safefall/`):

```
trials_shard_<s>.csv       one row per (trial, baseline)
scenarios_shard_<s>.jsonl  scenario spec + T/trigger/outcome
report/summary.csv         Table-IV-shaped overall summary
report/summary_by_mechanism.csv
report/paired_reductions.csv
report/metric_*.png/pdf    overall bars
report/mechanism_*.png/pdf per-mechanism bars
report/dist_*.png          distributions
report/index.html          gallery (plots + videos)
videos/trial_<id>_<baseline>.mp4
```

`TRIALS` is the total across shards; each shard takes `trial_id % shards == shard`.
`SHARDS` must match the Slurm array size.

**Spawn position.** Do not override the root position in `_reset`. The terrain term
broadcasts a single origin to every environment, so the default spawn is already
independent of env slot; an override that placed trial *i* on its own grid put trials
up to 252 m from the origin — off the plane — and they free-fell forever. Symptom:
a 10x drop in fall yield plus recorded heights of -8 to -40 m. Those heights were
correct measurements of robots below the floor, not corrupt data.

**`NUM_ENVS` is part of the reported configuration.** Fall dynamics are chaotic, so a
trial is not numerically reproducible across `NUM_ENVS` even though its scenario spec
and spawn state are keyed to `trial_id`. Reported numbers use **`NUM_ENVS=64`**, and
`NUM_ENVS` is recorded in each run's `manifest.json` — quote it with any result.
Smaller batches are valid for smoke tests and calibration only.

`--protected-patterns` defaults to `head,hand,wrist`. The benchmark aborts at startup
if no body matches, because an empty mask would silently report `illegal_contact = 0`
for every trial (that is exactly what happened before the pattern was made
robot-aware: the G1 model has no `head` or `hand` body).

## Important caveats

- **Check the force extraction before trusting numbers.** In `mujoco_warp`,
  `cfrc_ext`/`cfrc_int` are Warp `spatial_vector`s with the angular (torque) component
  first. The pre-existing `WarpBackend.compute_contact_forces()` returns
  `cfrc_t[..., :3]`, which is likely the **torque**, not the force. This benchmark uses
  the new explicit `[..., 3:6]` force path and per-contact forces, so its metrics are
  unaffected; the legacy `contact_forces` used by termination/rewards should be fixed
  separately with an A/B test.
- Absolute force/torque values depend on the contact model (`solref`/`solimp`),
  collision geometry, timestep, and friction. Reproduce relative baseline ordering and
  reductions first; absolute matching to the paper requires calibration.
- The paper's `J` (whole-body impulse) is ambiguous in the text; both max per-step and
  cumulative impulse are recorded.
- The paper's trigger lead is ~0.3-0.4 s (the 100 ms value is the predictor label
  boundary, not the activation time). The oracle uses `T - 300 ms` by default.
- The paper's failure set includes sensor noise, system delay, and dynamics mismatch;
  this benchmark implements the physical factors (push/slip/trip/freeze) first.
- Recommended pilot checks: per-foot force ≈ mg/2 at rest, baseline ordering
  `nominal > default ≳ damping` on joint torque/joint reaction/illegal contacts, and
  classic-vs-warp agreement on a handful of trials.
