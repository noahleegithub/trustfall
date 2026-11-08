#!/usr/bin/env python3
"""SafeFall baseline benchmark.

Runs walking + perturbation fall trials on a trained nominal policy and evaluates
the paper's three baselines from the same trigger state:

* ``nominal``  -- the locomotion policy continues through the fall
* ``default``  -- hold the default joint targets with nominal PD gains
* ``damping``  -- Kp ~= 0, Kd = 10, constant target

The fall is induced by 1-3 of: torso push, friction slip, foot trip, and actuator
failure (motor-off or brake-lock).  The trip factor has two modes: ``--trip-mode
obstacle`` (default) places a real box in front of the robot at the trigger so the foot
physically catches on it, and ``--trip-mode force`` applies a horizontal force to the
swing foot (the original synthetic trip, kept for reproducing older runs).  The impact
window starts at ``T - 300 ms`` (oracle trigger), where ``T`` is the first non-foot
ground contact after the perturbation.  Metrics are the SafeFall Table IV metrics
implemented in ``holosoma.simulator.mujoco.impact_metrics``.

Usage (single shard, pilot)::

    python -m scripts.safefall.benchmark \
        --checkpoint /scratch/.../model_025000.pt \
        --output-dir /scratch/.../safefall_eval \
        --trials 200 --num-envs 64 --seed 0 --shard 0 --num-shards 1

See ``scripts/safefall/bench.sh`` for the Slurm wrapper.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from holosoma.config_types.plugin import CameraVizPluginConfig
from holosoma.config_types.randomization import RandomizationManagerCfg
from holosoma.config_types.simulator import MujocoWarpConfig
from holosoma.config_types.termination import TerminationManagerCfg
from holosoma.config_types.terrain import ObstacleCfg, with_obstacle
from holosoma.config_values.logger import disabled as disabled_logger
from holosoma.config_values.sensor import CAMERA_REGISTRY
from holosoma.config_values.simulator import mujoco as mujoco_preset
from holosoma.config_values.simulator import mjwarp as mjwarp_preset
from holosoma.simulator.mujoco.impact_metrics import ImpactMonitor
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment

try:  # ``python -m scripts.safefall.benchmark`` (repo root on sys.path)
    from scripts.safefall.trip_placement import plan_trip_obstacle
except ModuleNotFoundError:  # the diag_*.py scripts add this directory to sys.path and import this module flat
    from trip_placement import plan_trip_obstacle

MECHANISMS = ("push", "slip", "trip", "freeze")
BASELINES = ("nominal", "default", "damping")
TRIP_MODES = ("obstacle", "force")

# Perturbation profiles. Measured fall yields on the G1 (200 trials, 64 envs, seed 0,
# corrected spawn): ``paper`` 26%, ``calibrated`` 62%.
#
# ``calibrated`` was raised while chasing an apparent 3-5% yield that turned out to be
# a spawn bug (trials were being placed off the ground plane), so the premise for
# raising strength was wrong. ``paper`` is the default because it reproduces the
# write-up's regime; ``calibrated`` is kept as an explicit stress profile.
# The active profile is recorded in the run manifest -- quote it with results.
PERTURBATION_PRESETS: dict[str, dict[str, float]] = {
    "paper": {
        "push_min": 1.5, "push_max": 3.5,
        "slip_min": 0.02, "slip_max": 0.15,
        "slip_duration_min": 0.10, "slip_duration_max": 0.30,
        "trip_force_min": 150.0, "trip_force_max": 400.0, "trip_duration": 0.05,
        "freeze_motor_off_p": 0.5, "freeze_two_joint_p": 0.3,
    },
    "calibrated": {
        # push: top of the paper range already fell ~85%; shift the whole band up.
        "push_min": 2.5, "push_max": 4.5,
        # slip: longer and slipper -- short, mild slips were the weakest mechanism.
        "slip_min": 0.01, "slip_max": 0.08,
        "slip_duration_min": 0.20, "slip_duration_max": 0.50,
        # trip: ~1.8x the force over twice the window.
        "trip_force_min": 400.0, "trip_force_max": 800.0, "trip_duration": 0.10,
        # freeze: motor-off failed ~3x more often than brake-lock; bias toward it
        # and toward disabling two joints.
        "freeze_motor_off_p": 0.8, "freeze_two_joint_p": 0.55,
    },
}

METRIC_FIELDS = (
    "max_joint_torque",
    "max_torque_ratio",
    "max_joint_reaction",
    "max_contact_force",
    "max_impulse",
    "cumulative_impulse",
    "illegal_contact",
    "torso_contact",
    # per-part severity, attributed by geom (head capsule is welded to torso_link)
    "peak_force_head",
    "peak_force_torso",
    "peak_force_hand",
    "cum_impulse_head",
    "cum_impulse_torso",
    "cum_impulse_hand",
    "max_limit_violations",
)


@dataclass
class Scenario:
    trial_id: int
    seed: int
    mechanisms: list[str]
    command: tuple[float, float, float]
    t_pert: int
    push_dv: tuple[float, float, float] | None = None
    slip_mu: float | None = None
    slip_steps: int = 0
    trip_force: float | None = None
    trip_steps: int = 0
    # Obstacle-mode trip: sampled top-face height at build time, direction/distance
    # resolved at the trigger from the robot's live motion. All None in force mode.
    trip_height: float | None = None
    trip_distance: float | None = None
    trip_dir_x: float | None = None
    trip_dir_y: float | None = None
    freeze_mode: str | None = None
    freeze_dofs: list[int] = field(default_factory=list)

    @property
    def primary(self) -> str:
        return self.mechanisms[0]


@dataclass
class PreFallResult:
    valid: torch.Tensor
    recovery: torch.Tensor
    t_step: torch.Tensor
    trigger_step: torch.Tensor
    post_end: torch.Tensor
    nominal: dict[str, torch.Tensor]
    trigger_root: torch.Tensor
    trigger_dof: torch.Tensor
    trigger_cmd: torch.Tensor
    trigger_friction: torch.Tensor
    trigger_freeze_mode: torch.Tensor
    trigger_freeze_mask: torch.Tensor
    trigger_freeze_q: torch.Tensor
    trigger_slip_remaining: torch.Tensor
    trigger_trip_remaining: torch.Tensor
    trigger_trip_force: torch.Tensor
    trigger_trip_body: torch.Tensor
    friction_baseline: torch.Tensor
    # Perturbation diagnostics: what was applied and what the body did afterwards.
    # These exist to explain a fall-vs-recovery outcome, not to score a policy.
    diag_push_norm: torch.Tensor
    diag_z_at_pert: torch.Tensor
    diag_z_min_post: torch.Tensor


class FallBenchmark:
    """Build the environment/policy once, then run batches of fall scenarios."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(args.output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.obstacle_cfg: ObstacleCfg | None = None
        self.obstacle_body_id: int | None = None
        self.obstacle_half_height: float | None = None
        self.obstacle_park_z: float | None = None
        self._build_config()
        self._build_env()
        self._setup_obstacle()
        self._setup_monitor()
        self._install_freeze_override()

    # ------------------------------------------------------------------ setup

    def _build_config(self) -> None:
        saved_cfg, wandb_path = load_saved_experiment_config(CheckpointConfig(checkpoint=self.args.checkpoint))
        base = saved_cfg.get_eval_config()
        render = self.args.render_trial_id >= 0
        num_envs = 1 if render else self.args.num_envs
        overrides = {
            "sensor": {},
            "plugin": {},
        }
        # Switch the backend but keep the checkpoint's physics/control rates.
        # The batch path uses mjwarp (GPU). Single-env rendering defaults to the classic
        # backend: mjwarp asserts device-side at num_envs=1 (the failure surfaces in
        # get_geom_friction), and one env gains nothing from GPU batching anyway.
        use_classic = render and self.args.render_backend == "classic"
        if use_classic:
            sim_preset = dataclasses.replace(
                mujoco_preset,
                config=dataclasses.replace(mujoco_preset.config, sim=base.simulator.config.sim),
            )
        else:
            sim_preset = dataclasses.replace(
                mjwarp_preset,
                config=dataclasses.replace(
                    mjwarp_preset.config,
                    sim=base.simulator.config.sim,
                    mujoco_warp=MujocoWarpConfig(nconmax_per_env=self.args.nconmax_per_env),
                ),
            )
        cfg = dataclasses.replace(base, simulator=sim_preset, logger=disabled_logger, **overrides)
        # No termination: the benchmark needs the full fall and post-impact window.
        # No domain randomization: evaluation scenarios vary the fall, not the robot
        # parameters, and this makes all baselines exactly paired.
        cfg = dataclasses.replace(
            cfg,
            termination=TerminationManagerCfg(terms={}),
            randomization=RandomizationManagerCfg(),
            training=dataclasses.replace(
                cfg.training,
                num_envs=num_envs,
                headless=True,
                seed=self.args.seed,
            ),
        )
        if self.args.trip_mode == "obstacle":
            # A real trip hazard instead of the synthetic foot force. Only this mode adds
            # the body, so force mode compiles exactly the model it always did.
            self.obstacle_cfg = ObstacleCfg()
            cfg = dataclasses.replace(cfg, terrain=with_obstacle(cfg.terrain, self.obstacle_cfg))
        if render:
            cfg = dataclasses.replace(cfg, sensor={"cam": CAMERA_REGISTRY[self.args.render_camera]})
        else:
            # No camera rendering in the batch benchmark.
            cfg = dataclasses.replace(cfg, sensor={})
        self.config = cfg
        self.saved_wandb_path = wandb_path
        (self.output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "checkpoint": self.args.checkpoint,
                    "wandb_path": wandb_path,
                    "args": vars(self.args),
                    "git_commit": _git_commit(),
                    "sif": _env("SIF"),
                },
                indent=2,
                default=str,
            )
        )

    def _build_env(self) -> None:
        self.env, self.device, self.simulation_app = setup_simulation_environment(self.config, device=self.device)
        self.sim = self.env.simulator
        self.control_hz = 1.0 / float(self.env.dt)
        self.ndof = int(self.env.num_dof)
        self.nbody = int(self.sim.root_model.nbody)
        self.num_envs = int(self.env.num_envs)

        algo_cls = get_class(self.config.algo._target_)
        if "ppo" not in str(self.config.algo._target_).lower():
            raise RuntimeError(
                f"This benchmark currently supports PPO checkpoints, got {self.config.algo._target_}. "
                "The SafeFall paper uses PPO."
            )
        algo = algo_cls(
            device=self.device,
            env=self.env,
            config=self.config.algo.config,
            log_dir=str(self.output_dir / "algo"),
            multi_gpu_cfg=None,
        )
        algo.setup()
        ckpt_path = load_checkpoint(self.args.checkpoint, str(self.output_dir / "checkpoint"))
        algo.load(str(ckpt_path))
        algo._eval_mode()
        algo.eval_policy = algo.get_inference_policy()
        self.algo = algo

        self.env.set_is_evaluating()
        self._zeros_actions = torch.zeros(self.num_envs, self.ndof, device=self.device)

    def _setup_obstacle(self) -> None:
        """Resolve the trip-hazard body id and its half height (no-op in force mode)."""
        if self.args.trip_mode != "obstacle":
            return
        cfg = self.obstacle_cfg
        if cfg is None:  # _build_config always sets it in obstacle mode; guard the invariant
            raise RuntimeError("trip mode is 'obstacle' but no ObstacleCfg was injected")
        import mujoco

        body_id = int(mujoco.mj_name2id(self.sim.root_model, mujoco.mjtObj.mjOBJ_BODY, cfg.name))
        if body_id < 0:
            raise RuntimeError(
                f"trip mode 'obstacle' needs a model body named '{cfg.name}', but the compiled "
                "model has none. The terrain builder only adds it when TerrainTermCfg.obstacle "
                "is set and the terrain mesh type is not 'none'/'fake'."
            )
        self.obstacle_body_id = body_id
        self.obstacle_half_height = float(cfg.half_extents[2])
        self.obstacle_park_z = float(cfg.pos[2])
        logger.info(
            f"Trip obstacle '{cfg.name}' -> body id {body_id}, "
            f"half_extents={tuple(float(v) for v in cfg.half_extents)}, parked at z={self.obstacle_park_z}"
        )

    def _setup_monitor(self) -> None:

        model = self.sim.root_model
        foot_body_ids = [
            int(self.sim.body_ids[i])
            for i, name in enumerate(self.sim.body_names)
            if "foot" in name.lower() or "ankle" in name.lower()
        ]
        if not foot_body_ids:
            raise RuntimeError("No foot/ankle bodies found; cannot identify ground impact bodies.")
        self.friction_geom_ids = [g for g in range(int(model.ngeom)) if int(model.geom_bodyid[g]) in set(foot_body_ids)]
        if not self.friction_geom_ids:
            raise RuntimeError("No geoms attached to foot/ankle bodies; cannot model friction slip.")
        # MuJoCo combines contact friction with ``max(friction1, friction2)``, so lowering only
        # the foot is ineffective when the ground is grippier.  Include every non-robot geom
        # (world/terrain) in the slip write; this simulates a slippery patch under the robot.
        robot_bodies = {int(b) for b in self.sim.body_ids}
        ground_geom_ids = [g for g in range(int(model.ngeom)) if int(model.geom_bodyid[g]) not in robot_bodies]
        self.friction_geom_ids = list(dict.fromkeys(self.friction_geom_ids + ground_geom_ids))
        self.foot_body_ids = foot_body_ids
        self.foot_holosoma_ids = [
            i for i, name in enumerate(self.sim.body_names) if "foot" in name.lower() or "ankle" in name.lower()
        ]

        window = max(48, self.args.post_steps + 8)
        self.base_friction = self.sim.get_geom_friction(
            self.friction_geom_ids, torch.arange(self.num_envs, device=self.device)
        )
        patterns = tuple(p.strip().lower() for p in self.args.protected_patterns.split(",") if p.strip())
        self.monitor = ImpactMonitor(self.env, self.sim, window_steps=window, protected_patterns=patterns)
        self.monitor.reset()
        n_protected = int(self.monitor.protected_mask.sum())
        if n_protected == 0:
            raise RuntimeError(
                "No bodies matched --protected-patterns "
                f"({', '.join(patterns)}); illegal_contact would be identically zero. "
                f"Available bodies: {', '.join(str(n) for n in self.sim.body_names)}"
            )
        logger.info(f"Protected bodies for illegal_contact ({n_protected}): {', '.join(patterns)}")
        n_head = int((self.monitor.geom_part == self.monitor.PARTS.index("head")).sum())
        if n_head == 0:
            raise RuntimeError(
                "No geom classified as 'head'; SafeFall head-impact severity would be "
                "identically zero. Add the head capsule name to the part patterns."
            )
        logger.info(f"Part geoms: head={n_head}")

    def _install_freeze_override(self) -> None:
        self.freeze_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.freeze_mask = torch.zeros(self.num_envs, self.ndof, dtype=torch.bool, device=self.device)
        self.freeze_q = torch.zeros(self.num_envs, self.ndof, device=self.device)
        self._kp_lock = float(self.args.brake_kp)
        self._kd_lock = float(self.args.brake_kd)

        term = self._joint_term()
        original = self.env._apply_force_in_physics_step

        def patched() -> None:
            original()
            if not bool(self.freeze_mode.any()):
                return
            q = self.sim.dof_pos[:]
            qd = self.sim.dof_vel[:]
            tau = term.torques
            motor = self.freeze_mode == 1
            brake = self.freeze_mode == 2
            if bool(motor.any()):
                tau[motor] = torch.where(self.freeze_mask[motor], torch.zeros_like(tau[motor]), tau[motor])
            if bool(brake.any()):
                desired = self._kp_lock * (self.freeze_q - q) - self._kd_lock * qd
                tau[brake] = torch.where(self.freeze_mask[brake], desired[brake], tau[brake])
            tau.clamp_(-self.env.torque_limits, self.env.torque_limits)
            self.sim.apply_torques_at_dof(tau)

        self.env._apply_force_in_physics_step = patched

    def _joint_term(self):
        for _, term in self.env.action_manager.iter_terms():
            if hasattr(term, "p_gains") and hasattr(term, "torques"):
                return term
        raise RuntimeError("No joint_control term with PD gains found.")

    # ------------------------------------------------------------------ scenarios

    def make_scenario(self, trial_id: int) -> Scenario:
        args = self.args
        rng = np.random.default_rng(args.seed * 100003 + trial_id * 7919 + 17)
        primary = MECHANISMS[trial_id % len(MECHANISMS)]
        n_factors = int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
        mechanisms = [primary]
        remaining = [m for m in MECHANISMS if m != primary]
        if n_factors > 1:
            k = min(n_factors - 1, len(remaining))
            mechanisms.extend(rng.choice(remaining, size=k, replace=False).tolist())

        command = (
            float(rng.uniform(0.5, 1.5)),
            float(rng.uniform(-0.4, 0.4)),
            float(rng.uniform(-0.5, 0.5)),
        )
        warmup_s = float(rng.uniform(args.warmup_min, args.warmup_max))
        t_pert = round(warmup_s * self.control_hz)
        t_pert = max(1, min(t_pert, args.max_pre_steps - 60))

        scenario = Scenario(
            trial_id=trial_id,
            seed=int(args.seed + trial_id),
            mechanisms=mechanisms,
            command=command,
            t_pert=t_pert,
        )
        if "push" in mechanisms:
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            mag = float(rng.uniform(args.push_min, args.push_max))
            scenario.push_dv = (mag * math.cos(angle), mag * math.sin(angle), 0.0)
        if "slip" in mechanisms:
            scenario.slip_mu = float(rng.uniform(args.slip_min, args.slip_max))
            scenario.slip_steps = max(
                1, round(rng.uniform(args.slip_duration_min, args.slip_duration_max) * self.control_hz)
            )
        if "trip" in mechanisms:
            if args.trip_mode == "force":
                # Unchanged draw order: force mode must keep reproducing old scenarios.
                scenario.trip_force = float(rng.uniform(args.trip_force_min, args.trip_force_max))
                scenario.trip_steps = max(1, round(args.trip_duration * self.control_hz))
            else:
                # Obstacle mode: the height is sampled now; direction and distance need the
                # robot's live motion, so they are resolved at the trigger (_apply_factors).
                scenario.trip_height = float(rng.uniform(args.trip_height_min, args.trip_height_max))
        if "freeze" in mechanisms:
            scenario.freeze_mode = str(
                rng.choice(["motor_off", "brake_lock"], p=[args.freeze_motor_off_p, 1.0 - args.freeze_motor_off_p])
            )
            leg_idx = [
                i for i, n in enumerate(self.sim.dof_names) if any(k in n.lower() for k in ("hip", "knee", "ankle"))
            ]
            arm_idx = [
                i
                for i, n in enumerate(self.sim.dof_names)
                if any(k in n.lower() for k in ("shoulder", "elbow", "wrist"))
            ]
            p2 = float(args.freeze_two_joint_p)
            n_joints = int(rng.choice([1, 2], p=[1.0 - p2, p2]))
            pool = leg_idx if (not arm_idx or rng.random() < 0.9) else arm_idx
            scenario.freeze_dofs = sorted(rng.choice(pool, size=min(n_joints, len(pool)), replace=False).tolist())
        return scenario

    # ------------------------------------------------------------------ rollout

    def _reset(self, commands: torch.Tensor, trial_ids: list[int]) -> dict:
        # Start every batch from a clean model/controller state.
        self.freeze_mode.zero_()
        self.freeze_mask.zero_()
        self._clear_all_applied()
        self.sim.set_geom_friction(
            self.friction_geom_ids, self.base_friction, torch.arange(self.num_envs, device=self.device)
        )
        term = self._joint_term()
        term._kp_scale[:] = 1.0
        term._kd_scale[:] = 1.0

        self.env.reset_all()
        # Park the trip hazard below the floor: a batch without a trip, and every non-trip
        # trial in a mixed batch, must see an unmodified scene. (reset_all resets data, not
        # model geometry, so this is the only place the box is cleared.)
        self._park_obstacle()
        # Do NOT override the root position. The terrain term broadcasts ONE origin to
        # every env (locomotion.py: `self._env_origins[:] = origin_0_0`), so the default
        # spawn is already independent of env slot. An earlier version placed trial i at
        # (i%64*4, i//64*4) -- up to 252 m away, i.e. off the ground plane -- and those
        # trials free-fell forever: fall yield collapsed 26% -> 3% and the "height"
        # diagnostics read -8..-40 m because that is where the robots actually were.
        k = int(commands.shape[0])
        ids = torch.arange(k, device=self.device, dtype=torch.long)
        dof = torch.zeros(self.num_envs, self.ndof, 2, device=self.device)
        base_pos = self.env.default_dof_pos.to(self.device)
        # Mirror the env's own reset distribution -- joint_pos = default * U(0.5, 1.5),
        # joint_vel = 0 -- but draw it per trial so it is reproducible and independent
        # of batch layout (the env draws it from the global RNG).
        for i, tid in enumerate(trial_ids):
            rng = np.random.default_rng(self.args.seed * 100003 + int(tid) * 7919 + 41)
            scale = torch.as_tensor(
                rng.uniform(0.5, 1.5, self.ndof), dtype=torch.float32, device=self.device
            )
            dof[i, :, 0] = base_pos[i] * scale
        self.sim.set_dof_state_tensor_robots(ids, dof.reshape(-1, 2))
        self.env.command_manager.commands[: commands.shape[0]] = commands.to(self.device)
        self.env.observation_manager.reset()
        self.env._compute_observations()
        self.env._clip_observations()
        self.monitor.reset()
        return self.env.obs_buf_dict

    def _clear_all_applied(self) -> None:
        af = self.sim.applied_forces
        if hasattr(af, "zero_"):
            af.zero_()
        else:
            af.fill(0.0)

    def _park_obstacle(self) -> None:
        """Sink the trip obstacle below the ground plane for every env (no-op without one)."""
        if self.obstacle_body_id is None or self.obstacle_park_z is None:
            return
        n = self.num_envs
        positions = torch.zeros(n, 1, 3, device=self.device)
        positions[:, 0, 2] = self.obstacle_park_z
        quats = torch.zeros(n, 1, 4, device=self.device)
        quats[:, 0, 3] = 1.0  # xyzw identity
        self.sim.backend.set_static_body_world_pose(
            [self.obstacle_body_id],
            positions,
            quats,
            env_ids=torch.arange(n, device=self.device, dtype=torch.long),
        )

    def _actor_obs(self, obs: dict) -> torch.Tensor:
        return torch.cat([obs[k] for k in self.algo.actor_obs_keys], dim=1)

    def _policy_actions(self, obs: dict) -> torch.Tensor:
        return self.algo.eval_policy({"actor_obs": self._actor_obs(obs)})

    def _dof_state_3d(self) -> torch.Tensor:
        ds = self.sim.dof_state
        shape = tuple(ds.shape)
        arr = ds[:] if len(shape) == 2 else ds
        if len(shape) == 2 and shape[0] == self.num_envs * self.ndof:
            return arr.view(self.num_envs, self.ndof, 2)
        return arr.view(1, self.ndof, 2)

    def _set_applied(self, env_ids: torch.Tensor, body_ids: torch.Tensor, forces: torch.Tensor) -> None:
        af = self.sim.applied_forces
        if af.ndim == 3:
            af[env_ids, body_ids, 0:3] = forces
        else:
            # Classic single-env backend (numpy view).
            af[body_ids[0].item(), 0:3] = forces[0].detach().cpu().numpy()

    def _clear_applied(self, env_ids: torch.Tensor, body_ids: torch.Tensor) -> None:
        zero = torch.zeros(len(env_ids), 3, device=self.device)
        self._set_applied(env_ids, body_ids, zero)

    def _apply_factors(
        self,
        scenarios: list[Scenario],
        env_ids: torch.Tensor,
        *,
        slip_end: torch.Tensor,
        slip_mu: torch.Tensor,
        slip_baseline: torch.Tensor,
        trip_end: torch.Tensor,
        trip_force: torch.Tensor,
        trip_body: torch.Tensor,
    ) -> None:
        """Apply the selected factors to ``env_ids`` (parallel to ``scenarios``)."""
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        # push
        push_local = [j for j, s in enumerate(scenarios) if s.push_dv is not None]
        if push_local:
            idx = env_ids[push_local]
            root = self.sim.robot_root_states.clone()
            dv = torch.tensor([scenarios[j].push_dv for j in push_local], device=self.device, dtype=root.dtype)
            root[idx, 7:10] += dv
            self.sim.set_actor_root_state_tensor_robots(idx, root[idx])

        # friction slip
        slip_local = [j for j, s in enumerate(scenarios) if s.slip_mu is not None]
        if slip_local:
            idx = env_ids[slip_local]
            base = self.sim.get_geom_friction(self.friction_geom_ids, idx)
            slip_baseline[idx] = base
            mu = torch.tensor([scenarios[j].slip_mu for j in slip_local], device=self.device, dtype=base.dtype)
            values = base.clone()
            values[:] = mu.unsqueeze(1)
            self.sim.set_geom_friction(self.friction_geom_ids, values, idx)
            steps = torch.tensor([scenarios[j].slip_steps for j in slip_local], device=self.device, dtype=torch.long)
            t_pert = torch.tensor([scenarios[j].t_pert for j in slip_local], device=self.device, dtype=torch.long)
            slip_end[idx] = t_pert + steps
            slip_mu[idx] = mu

        # trip (obstacle mode): put the box in front of the robot's current motion so the
        # swing foot physically catches on it. Obstacle-mode scenarios carry no trip_force,
        # so the force block below skips them; the force block is untouched otherwise.
        obstacle_local = [j for j, s in enumerate(scenarios) if s.trip_height is not None]
        if obstacle_local and self.obstacle_body_id is not None and self.obstacle_half_height is not None:
            idx = env_ids[obstacle_local]
            # A clone is required: the root_states proxy view misbehaves with 2-D indexing.
            root = self.sim.robot_root_states.clone()
            plan = plan_trip_obstacle(
                base_xy=root[idx, 0:2].detach().cpu().numpy(),
                velocity_xy=root[idx, 7:9].detach().cpu().numpy(),
                command_xy=np.asarray([scenarios[j].command[0:2] for j in obstacle_local], dtype=np.float64),
                trip_height=np.asarray([scenarios[j].trip_height for j in obstacle_local], dtype=np.float64),
                half_height=self.obstacle_half_height,
                lead_time=float(self.args.trip_lead_time),
                min_distance=float(self.args.trip_min_distance),
                max_distance=float(self.args.trip_max_distance),
            )
            for k, j in enumerate(obstacle_local):
                scenario = scenarios[j]
                scenario.trip_distance = float(plan.distance[k])
                scenario.trip_dir_x = float(plan.direction[k, 0])
                scenario.trip_dir_y = float(plan.direction[k, 1])
            self.sim.backend.set_static_body_world_pose(
                [self.obstacle_body_id],
                torch.as_tensor(plan.center[:, None, :], device=self.device, dtype=torch.float32),
                torch.as_tensor(plan.quat[:, None, :], device=self.device, dtype=torch.float32),
                env_ids=idx,
            )

        # trip: push the swing foot (lower normal force)
        trip_local = [j for j, s in enumerate(scenarios) if s.trip_force is not None]
        if trip_local:
            idx = env_ids[trip_local]
            foot_holo = torch.as_tensor(self.foot_holosoma_ids, device=self.device, dtype=torch.long)
            foot_raw = torch.as_tensor(self.foot_body_ids, device=self.device, dtype=torch.long)
            normal = self.sim.contact_forces[:, foot_holo, 2]
            swing_local = normal[idx].argmin(dim=1)
            body = foot_raw[swing_local]
            # Paired advanced indexing: gather (env, its swing foot) in one step.
            # Chaining [idx][:, foot_ids] would instead produce an (n, n, 2)
            # cross-product of every env against every foot.
            vel = self.sim._rigid_body_vel[idx, foot_holo[swing_local], :2]
            speed = torch.linalg.vector_norm(vel, dim=-1, keepdim=True)
            direction = -vel / speed.clamp_min(1e-6)
            # Trial-keyed fallback direction. A raw torch.randn here depends on how
            # many draws preceded it in the process, i.e. on batch composition.
            dirs = []
            for j in trip_local:
                trng = np.random.default_rng(self.args.seed * 100003 + scenarios[j].trial_id * 7919 + 29)
                ang = float(trng.uniform(0.0, 2.0 * math.pi))
                dirs.append((math.cos(ang), math.sin(ang)))
            random_dir = torch.tensor(dirs, device=self.device, dtype=torch.float32)
            direction = torch.where(speed > 0.1, direction, random_dir)
            mag = torch.tensor([scenarios[j].trip_force for j in trip_local], device=self.device, dtype=torch.float32)
            forces = torch.cat([direction * mag.unsqueeze(1), (0.15 * mag).unsqueeze(1)], dim=1)
            self._set_applied(idx, body, forces)
            trip_body[idx] = body
            trip_force[idx] = forces
            steps = torch.tensor([scenarios[j].trip_steps for j in trip_local], device=self.device, dtype=torch.long)
            t_pert = torch.tensor([scenarios[j].t_pert for j in trip_local], device=self.device, dtype=torch.long)
            trip_end[idx] = t_pert + steps

        # actuator failure
        for j, s in enumerate(scenarios):
            if s.freeze_mode is None:
                continue
            i = int(env_ids[j].item())
            self.freeze_mode[i] = 1 if s.freeze_mode == "motor_off" else 2
            self.freeze_mask[i] = False
            if s.freeze_dofs:
                self.freeze_mask[i, s.freeze_dofs] = True
            self.freeze_q[i] = self.sim.dof_pos[i].clone()

        # First non-foot ground contact is now meaningful (ignore walking contacts).
        self.monitor.clear_trial_start(env_ids)

    def run_prefall(self, scenarios: list[Scenario], *, nominal_metrics: bool = True) -> PreFallResult:
        n = len(scenarios)
        args = self.args
        N = self.num_envs
        k_state = max(32, args.oracle_lead_steps + 4)

        commands = torch.tensor([s.command for s in scenarios], device=self.device, dtype=torch.float32)
        obs = self._reset(commands, [s.trial_id for s in scenarios])

        t_pert = torch.tensor([s.t_pert for s in scenarios], device=self.device, dtype=torch.long)
        phase = torch.zeros(n, dtype=torch.long, device=self.device)  # 0=pre,1=falling,2=post,3=done,4=recovery
        t_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        trigger_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        post_end = torch.full((n,), -1, dtype=torch.long, device=self.device)
        valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        recovery = torch.zeros(n, dtype=torch.bool, device=self.device)
        done = torch.zeros(n, dtype=torch.bool, device=self.device)

        trigger_root = torch.zeros(n, 13, device=self.device)
        trigger_dof = torch.zeros(n, self.ndof, 2, device=self.device)
        trigger_cmd = commands.clone()
        trigger_friction = torch.zeros(n, len(self.friction_geom_ids), device=self.device)
        trigger_freeze_mode = torch.zeros(n, dtype=torch.long, device=self.device)
        trigger_freeze_mask = torch.zeros(n, self.ndof, dtype=torch.bool, device=self.device)
        trigger_freeze_q = torch.zeros(n, self.ndof, device=self.device)
        trigger_slip_remaining = torch.zeros(n, dtype=torch.long, device=self.device)
        trigger_trip_remaining = torch.zeros(n, dtype=torch.long, device=self.device)
        trigger_trip_force = torch.zeros(n, 3, device=self.device)
        trigger_trip_body = torch.zeros(n, dtype=torch.long, device=self.device)

        slip_end = torch.full((n,), -1, dtype=torch.long, device=self.device)
        slip_mu = torch.zeros(n, device=self.device)
        slip_baseline = torch.zeros(n, len(self.friction_geom_ids), device=self.device)
        trip_end = torch.full((n,), -1, dtype=torch.long, device=self.device)
        trip_force = torch.zeros(n, 3, device=self.device)
        trip_body = torch.zeros(n, dtype=torch.long, device=self.device)

        state_root = torch.zeros(k_state, N, 13, device=self.device)
        state_dof = torch.zeros(k_state, N, self.ndof, 2, device=self.device)
        state_root[0] = self.sim.robot_root_states.clone()
        state_dof[0] = self._dof_state_3d().clone()
        slip_baseline[:] = self.sim.get_geom_friction(self.friction_geom_ids, torch.arange(n, device=self.device))

        nominal: dict[str, torch.Tensor] = {key: torch.zeros(n, device=self.device) for key in METRIC_FIELDS}

        # Perturbation diagnostics.
        diag_push_norm = torch.zeros(n, device=self.device)
        diag_z_at_pert = torch.full((n,), -1.0, device=self.device)
        diag_z_min_post = torch.full((n,), float("inf"), device=self.device)

        max_steps = args.max_pre_steps + args.post_steps
        s = 1
        while s <= max_steps:
            state_root[s % k_state] = self.sim.robot_root_states.clone()
            state_dof[s % k_state] = self._dof_state_3d().clone()

            # Start perturbations / restore slip and trip.
            start_ids = ((phase == 0) & (s == t_pert)).nonzero(as_tuple=False).flatten()
            if start_ids.numel() > 0:
                started = [scenarios[i] for i in start_ids.tolist()]
                diag_push_norm[start_ids] = torch.tensor(
                    [float(np.linalg.norm(sc.push_dv[:2])) if sc.push_dv else 0.0 for sc in started],
                    device=self.device,
                )
                # Read from the clone snapshotted at the top of this iteration, so the
                # value matches the state used for the trigger snapshot.
                diag_z_at_pert[start_ids] = state_root[s % k_state][start_ids, 2]
                self._apply_factors(
                    started,
                    start_ids,
                    slip_end=slip_end,
                    slip_mu=slip_mu,
                    slip_baseline=slip_baseline,
                    trip_end=trip_end,
                    trip_force=trip_force,
                    trip_body=trip_body,
                )
                phase[start_ids] = 1

            restore_slip = (slip_end == s).nonzero(as_tuple=False).flatten()
            if restore_slip.numel() > 0:
                self.sim.set_geom_friction(self.friction_geom_ids, slip_baseline[restore_slip], restore_slip)
                # Keep slip_end: a trigger detected at T > s still needs to know
                # whether slip was active at its (earlier) trigger step.
            clear_trip = (trip_end == s).nonzero(as_tuple=False).flatten()
            if clear_trip.numel() > 0:
                self._clear_applied(clear_trip, trip_body[clear_trip])
                # Keep trip_end/trip_force/trip_body for the same reason.

            actions = self._policy_actions(obs)
            obs, _, _, _ = self.env.step({"actions": actions, "obs": obs})

            # Track how low the body got after the perturbation: a trial that
            # "did not fall" should still show a measurable drop, and one that
            # barely moved points at the perturbation not being applied.
            z_now = self.sim.robot_root_states.clone()[:n, 2]
            post_pert = phase >= 1
            diag_z_min_post = torch.where(post_pert & (z_now < diag_z_min_post), z_now, diag_z_min_post)

            # Impact detection: T = first non-foot ground contact after perturbation.
            newly = (
                ((phase == 1) & (t_step < 0) & (self.monitor.first_contact_step[:n] >= 0))
                .nonzero(as_tuple=False)
                .flatten()
            )
            if newly.numel() > 0:
                tvals = self.monitor.first_contact_step[newly]
                trig = torch.maximum(t_pert[newly] + 1, tvals - self.args.oracle_lead_steps)
                trig = torch.minimum(trig, torch.full_like(trig, s))
                for j, i in enumerate(newly.tolist()):
                    ts = int(trig[j].item())
                    t_step[i] = tvals[j]
                    trigger_step[i] = ts
                    post_end[i] = ts + self.args.post_steps - 1
                    valid[i] = True
                    idx = ts % k_state
                    trigger_root[i] = state_root[idx, i]
                    trigger_dof[i] = state_dof[idx, i]
                    trigger_freeze_mode[i] = self.freeze_mode[i]
                    trigger_freeze_mask[i] = self.freeze_mask[i]
                    trigger_freeze_q[i] = self.freeze_q[i]
                    slip_end_i = int(slip_end[i].item())
                    trip_end_i = int(trip_end[i].item())
                    trigger_slip_remaining[i] = max(0, slip_end_i - ts) if slip_end_i >= 0 else 0
                    trigger_trip_remaining[i] = max(0, trip_end_i - ts) if trip_end_i >= 0 else 0
                    trigger_trip_force[i] = trip_force[i]
                    trigger_trip_body[i] = trip_body[i]
                    if slip_end_i >= 0 and ts < slip_end_i:
                        trigger_friction[i] = slip_mu[i]
                    else:
                        trigger_friction[i] = slip_baseline[i]
                phase[newly] = 2

            # Snapshot completed nominal windows.
            completed = ((phase == 2) & (s >= post_end) & ~done).nonzero(as_tuple=False).flatten()
            if completed.numel() > 0:
                if nominal_metrics:
                    res = self.monitor.results(completed, trigger_step[completed], post_end[completed])
                    for key in METRIC_FIELDS:
                        nominal[key][completed] = getattr(res, key).to(torch.float32)
                done[completed] = True
                phase[completed] = 3

            # Give up on scenarios that never fell.
            lost = ((phase == 0) | (phase == 1)) & (s >= self.args.max_pre_steps)
            if lost.any():
                recovery[lost] = True
                phase[lost] = 4

            if bool(((phase == 3) | (phase == 4)).all()):
                break
            s += 1

        return PreFallResult(
            valid=valid,
            recovery=recovery,
            t_step=t_step,
            trigger_step=trigger_step,
            post_end=post_end,
            nominal=nominal,
            trigger_root=trigger_root,
            trigger_dof=trigger_dof,
            trigger_cmd=trigger_cmd,
            trigger_friction=trigger_friction,
            trigger_freeze_mode=trigger_freeze_mode,
            trigger_freeze_mask=trigger_freeze_mask,
            trigger_freeze_q=trigger_freeze_q,
            trigger_slip_remaining=trigger_slip_remaining,
            trigger_trip_remaining=trigger_trip_remaining,
            trigger_trip_force=trigger_trip_force,
            trigger_trip_body=trigger_trip_body,
            friction_baseline=slip_baseline,
            diag_push_norm=diag_push_norm,
            diag_z_at_pert=diag_z_at_pert,
            diag_z_min_post=torch.where(
                torch.isfinite(diag_z_min_post), diag_z_min_post, torch.full_like(diag_z_min_post, -1.0)
            ),
        )

    def _prepare_branch(self, pre: PreFallResult, baseline: str) -> torch.Tensor:
        env_ids = pre.valid.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return env_ids
        n = len(pre.valid)
        all_ids = torch.arange(n, device=self.device)
        # Reset any residual perturbation/failure state before the env's internal
        # reset step, so it cannot leak into the branch.
        term = self._joint_term()
        self.freeze_mode.zero_()
        self.freeze_mask.zero_()
        self._clear_applied(all_ids, pre.trigger_trip_body)
        self.sim.set_geom_friction(self.friction_geom_ids, pre.friction_baseline, all_ids)
        term._kp_scale[:] = 1.0
        term._kd_scale[:] = 1.0

        self.env.reset_all()
        self.env.command_manager.commands[env_ids] = pre.trigger_cmd[env_ids]
        self.sim.set_actor_root_state_tensor_robots(env_ids, pre.trigger_root[env_ids])
        dof = self._dof_state_3d()
        dof[env_ids] = pre.trigger_dof[env_ids]
        self.sim.set_dof_state_tensor_robots(env_ids, dof.reshape(-1, 2))
        self.env.action_manager.reset(env_ids)
        self.env.observation_manager.reset(env_ids)

        # Restore friction at the trigger.
        self.sim.set_geom_friction(self.friction_geom_ids, pre.trigger_friction[env_ids], env_ids)
        # Restore actuator-failure state.
        self.freeze_mode[env_ids] = pre.trigger_freeze_mode[env_ids]
        self.freeze_mask[env_ids] = pre.trigger_freeze_mask[env_ids]
        self.freeze_q[env_ids] = pre.trigger_freeze_q[env_ids]
        # Restore any residual trip force.
        residual = pre.trigger_trip_remaining[env_ids] > 0
        if residual.any():
            rid = env_ids[residual]
            self._set_applied(rid, pre.trigger_trip_body[rid], pre.trigger_trip_force[rid])
        # Controller gains.
        if baseline == "damping":
            term._kp_scale[env_ids] = (1e-5 / term.p_gains.clamp_min(1e-6)).unsqueeze(0).expand(len(env_ids), -1)
            term._kd_scale[env_ids] = (10.0 / term.d_gains.clamp_min(1e-6)).unsqueeze(0).expand(len(env_ids), -1)
        else:
            term._kp_scale[env_ids] = 1.0
            term._kd_scale[env_ids] = 1.0
        self.monitor.reset()
        return env_ids

    def run_branch(self, pre: PreFallResult, baseline: str) -> dict[str, torch.Tensor]:
        args = self.args
        env_ids = self._prepare_branch(pre, baseline)
        n = len(pre.valid)
        out: dict[str, torch.Tensor] = {key: torch.zeros(n, device=self.device) for key in METRIC_FIELDS}
        if env_ids.numel() == 0:
            return out

        slip_left = pre.trigger_slip_remaining[env_ids].clone()
        trip_left = pre.trigger_trip_remaining[env_ids].clone()
        baseline_fric = pre.friction_baseline[env_ids]

        for s in range(1, args.post_steps + 1):
            restore = (slip_left == s - 1) & (slip_left > 0)
            if restore.any():
                rid = env_ids[restore]
                self.sim.set_geom_friction(self.friction_geom_ids, baseline_fric[restore], rid)
            clear = (trip_left == s - 1) & (trip_left > 0)
            if clear.any():
                cid = env_ids[clear]
                self._clear_applied(cid, pre.trigger_trip_body[cid])
            self.env.step({"actions": self._zeros_actions})

        res = self.monitor.results(env_ids, torch.ones_like(env_ids), torch.full_like(env_ids, args.post_steps))
        for key in METRIC_FIELDS:
            value = getattr(res, key).to(torch.float32)
            out[key][env_ids] = value

        # Cleanup so the next baseline starts from a nominal model.
        self.sim.set_geom_friction(self.friction_geom_ids, pre.friction_baseline[env_ids], env_ids)
        self._clear_applied(env_ids, pre.trigger_trip_body[env_ids])
        self.freeze_mode[env_ids] = 0
        term = self._joint_term()
        term._kp_scale[env_ids] = 1.0
        term._kd_scale[env_ids] = 1.0
        return out

    # ------------------------------------------------------------------ driver

    def render_trial(self) -> None:
        """Replay one scenario with the camera plugin and save an mp4.

        Two passes: first find ``T``/trigger (without rendering), then replay the
        full scenario from the start with the selected controller switch at the
        trigger.  The replay is deterministic (no domain randomization, fixed seed).
        """
        args = self.args
        if self.num_envs != 1:
            raise RuntimeError("--render-trial-id requires --num-envs 1")
        scenario = self.make_scenario(args.render_trial_id)
        logger.info(f"Render trial {scenario.trial_id}: {scenario.mechanisms} {args.render_baseline}")
        pre = self.run_prefall([scenario], nominal_metrics=False)
        if not bool(pre.valid[0]):
            raise RuntimeError(f"Trial {scenario.trial_id} did not fall; nothing to render.")

        # Clear anything left by the pre-fall pass.
        ids = torch.tensor([0], device=self.device, dtype=torch.long)
        term = self._joint_term()
        self.freeze_mode.zero_()
        self.freeze_mask.zero_()
        self._clear_applied(ids, pre.trigger_trip_body)
        self.sim.set_geom_friction(self.friction_geom_ids, pre.friction_baseline, ids)
        term._kp_scale[:] = 1.0
        term._kd_scale[:] = 1.0

        obs = self._reset(torch.tensor([scenario.command], device=self.device), [scenario.trial_id])
        self.monitor.reset()
        # Camera framing diagnostics: a world-mount camera sits at its compiled spec pose, while the
        # robot spawns at the terrain env origin. If those differ, an overview clip frames empty ground.
        try:
            origins = self.env.scene.env_origins
            root = self.sim.robot_root_states[0, :3]
            logger.info(f"RENDER FRAMING env_origin={origins[0].tolist()} robot_root={root.tolist()}")
        except Exception as exc:  # diagnostics must never break rendering
            logger.warning(f"RENDER FRAMING unavailable: {exc}")

        out_dir = Path(args.render_dir).expanduser() if args.render_dir else (self.output_dir / "videos")
        out_dir.mkdir(parents=True, exist_ok=True)
        video_cfg = CameraVizPluginConfig(
            record_video=True,
            save_dir=str(out_dir),
            env_ids=[0],
            cameras=["cam"],
            playback_rate=float(args.render_playback_rate),
        )
        video_cfg.get_cls()(video_cfg, self.sim)

        slip_end = torch.full((1,), -1, dtype=torch.long, device=self.device)
        slip_mu = torch.zeros(1, device=self.device)
        slip_baseline = self.sim.get_geom_friction(self.friction_geom_ids, ids)
        trip_end = torch.full((1,), -1, dtype=torch.long, device=self.device)
        trip_force = torch.zeros(1, 3, device=self.device)
        trip_body = torch.zeros(1, dtype=torch.long, device=self.device)

        trigger = int(pre.trigger_step[0])
        end = int(pre.post_end[0])
        switched = False
        for s in range(1, end + 1):
            if s == scenario.t_pert:
                self._apply_factors(
                    [scenario],
                    ids,
                    slip_end=slip_end,
                    slip_mu=slip_mu,
                    slip_baseline=slip_baseline,
                    trip_end=trip_end,
                    trip_force=trip_force,
                    trip_body=trip_body,
                )
            if slip_end[0] >= 0 and s == int(slip_end[0].item()):
                self.sim.set_geom_friction(self.friction_geom_ids, slip_baseline, ids)
                slip_end[0] = -1
            if trip_end[0] >= 0 and s == int(trip_end[0].item()):
                self._clear_applied(ids, trip_body)
                trip_end[0] = -1

            if not switched and s >= trigger and args.render_baseline != "nominal":
                if args.render_baseline == "damping":
                    term._kp_scale[0] = 1e-5 / term.p_gains.clamp_min(1e-6)
                    term._kd_scale[0] = 10.0 / term.d_gains.clamp_min(1e-6)
                else:
                    term._kp_scale[0] = 1.0
                    term._kd_scale[0] = 1.0
                switched = True

            if args.render_baseline == "nominal" or not switched:
                actions = self._policy_actions(obs)
            else:
                actions = self._zeros_actions
            obs, _, _, _ = self.env.step({"actions": actions, "obs": obs})

        self.env.simulator.close()  # CLOSE hook encodes the mp4
        # Rename the timestamped file to a stable, report-friendly name.
        videos = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if videos:
            target = out_dir / f"trial_{scenario.trial_id}_{args.render_baseline}.mp4"
            videos[-1].replace(target)
            logger.info(f"Wrote {target}")
        else:
            logger.warning("Camera plugin produced no video (check camera/sensor config).")

    def run(self) -> None:
        args = self.args
        trial_ids = list(range(args.shard, args.trials, args.num_shards))
        if args.limit:
            trial_ids = trial_ids[: args.limit]
        logger.info(f"Shard {args.shard}/{args.num_shards}: {len(trial_ids)} trials")

        csv_path = self.output_dir / f"trials_shard_{args.shard}.csv"
        jsonl_path = self.output_dir / f"scenarios_shard_{args.shard}.jsonl"
        write_header = not csv_path.exists()
        if not write_header:
            # Appending rows with a different column set silently corrupts the file
            # (report.py then either raises on field counts or misaligns columns).
            with csv_path.open(newline="") as fh:
                existing = next(csv.reader(fh), [])
            if existing != self._csv_fields():
                raise RuntimeError(
                    f"{csv_path} was written with a different schema "
                    f"({len(existing)} columns vs {len(self._csv_fields())}); "
                    "use a fresh output directory rather than appending."
                )
        with csv_path.open("a", newline="") as f_csv, jsonl_path.open("a") as f_json:
            writer = csv.DictWriter(f_csv, fieldnames=self._csv_fields())
            if write_header:
                writer.writeheader()

            b = 0
            while b < len(trial_ids):
                batch_ids = trial_ids[b : b + self.num_envs]
                scenarios = [self.make_scenario(tid) for tid in batch_ids]
                logger.info(f"Batch {b // self.num_envs}: trials {batch_ids[0]}..{batch_ids[-1]}")
                pre = self.run_prefall(scenarios)
                branches = {
                    "default": self.run_branch(pre, "default"),
                    "damping": self.run_branch(pre, "damping"),
                }

                for i, scenario in enumerate(scenarios):
                    row_meta = self._scenario_meta(scenario, pre, i)
                    f_json.write(json.dumps(row_meta) + "\n")
                    if not bool(pre.valid[i]):
                        continue
                    for baseline in BASELINES:
                        values = pre.nominal if baseline == "nominal" else branches[baseline]
                        row = {
                            "trial_id": scenario.trial_id,
                            "baseline": baseline,
                            "mechanisms": "+".join(scenario.mechanisms),
                            "primary": scenario.primary,
                            "command_vx": scenario.command[0],
                            "command_vy": scenario.command[1],
                            "command_yaw": scenario.command[2],
                            "t_pert_step": scenario.t_pert,
                            "t_step": int(pre.t_step[i]),
                            "trigger_step": int(pre.trigger_step[i]),
                            "post_end": int(pre.post_end[i]),
                            "push_dv_x": scenario.push_dv[0] if scenario.push_dv else 0.0,
                            "push_dv_y": scenario.push_dv[1] if scenario.push_dv else 0.0,
                            "slip_mu": scenario.slip_mu or 0.0,
                            "trip_force": scenario.trip_force or 0.0,
                            "trip_mode": args.trip_mode,
                            "trip_height": scenario.trip_height or 0.0,
                            "trip_distance": scenario.trip_distance or 0.0,
                            "trip_dir_x": scenario.trip_dir_x or 0.0,
                            "trip_dir_y": scenario.trip_dir_y or 0.0,
                            "freeze_mode": scenario.freeze_mode or "",
                            "freeze_dofs": ";".join(map(str, scenario.freeze_dofs)),
                        }
                        row["push_dv_norm"] = float(pre.diag_push_norm[i])
                        row["root_z_at_pert"] = float(pre.diag_z_at_pert[i])
                        row["root_z_min_post"] = float(pre.diag_z_min_post[i])
                        for key in METRIC_FIELDS:
                            row[key] = float(values[key][i].item())
                        parts = {p: row[f"peak_force_{p}"] for p in ("head", "torso", "hand")}
                        worst = max(parts, key=lambda k: parts[k])
                        row["worst_hit_part"] = worst if parts[worst] > 0.0 else ""
                        writer.writerow(row)
                f_csv.flush()
                b += len(batch_ids)

        logger.info(f"Wrote {csv_path} and {jsonl_path}")

    @staticmethod
    def _csv_fields() -> list[str]:
        return [
            "trial_id",
            "baseline",
            "mechanisms",
            "primary",
            "command_vx",
            "command_vy",
            "command_yaw",
            "t_pert_step",
            "t_step",
            "trigger_step",
            "post_end",
            "push_dv_x",
            "push_dv_y",
            "slip_mu",
            "trip_force",
            "trip_mode",
            "trip_height",
            "trip_distance",
            "trip_dir_x",
            "trip_dir_y",
            "freeze_mode",
            "freeze_dofs",
            "push_dv_norm",
            "root_z_at_pert",
            "root_z_min_post",
            "worst_hit_part",
            *METRIC_FIELDS,
        ]

    def _scenario_meta(self, scenario: Scenario, pre: PreFallResult, i: int) -> dict:
        return {
            "trial_id": scenario.trial_id,
            "seed": scenario.seed,
            "mechanisms": scenario.mechanisms,
            "primary": scenario.primary,
            "command": scenario.command,
            "t_pert_step": scenario.t_pert,
            "push_dv": scenario.push_dv,
            "slip_mu": scenario.slip_mu,
            "slip_steps": scenario.slip_steps,
            "trip_force": scenario.trip_force,
            "trip_steps": scenario.trip_steps,
            "trip_mode": self.args.trip_mode,
            "trip_height": scenario.trip_height,
            "trip_distance": scenario.trip_distance,
            "trip_dir_x": scenario.trip_dir_x,
            "trip_dir_y": scenario.trip_dir_y,
            "freeze_mode": scenario.freeze_mode,
            "freeze_dofs": scenario.freeze_dofs,
            "t_step": int(pre.t_step[i]),
            "trigger_step": int(pre.trigger_step[i]),
            "post_end": int(pre.post_end[i]),
            "valid": bool(pre.valid[i]),
            "recovery": bool(pre.recovery[i]),
            # Perturbation diagnostics live here, not only in the CSV: a trial that
            # recovers writes no CSV rows, so the CSV alone cannot answer "did the
            # perturbation actually move the robot?" for the trials that didn't fall.
            "push_dv_norm": float(pre.diag_push_norm[i]),
            "root_z_at_pert": float(pre.diag_z_at_pert[i]),
            "root_z_min_post": float(pre.diag_z_min_post[i]),
        }

    def close(self) -> None:
        try:
            self.env.simulator.close()
        finally:
            if self.simulation_app:
                close_simulation_app(self.simulation_app)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _env(name: str) -> str | None:
    return os.environ.get(name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SafeFall baseline fall benchmark")
    p.add_argument("--checkpoint", required=True, help="Local .pt or wandb://... checkpoint of the nominal policy")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--trials", type=int, default=200, help="Total trials across all shards")
    p.add_argument("--limit", type=int, default=0, help="Optional cap on this shard's trials (for debugging)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--nconmax-per-env", type=int, default=192, help="mjwarp contact capacity per env")
    p.add_argument(
        "--protected-patterns",
        default="head,hand,wrist",
        help="Comma-separated body-name substrings counted as protected links for "
        "illegal_contact. Defaults suit a headless humanoid: wrists stand in for hands. "
        "Torso contact is tracked separately as torso_contact, not counted here, because "
        "every fall lands on the torso and would saturate illegal_contact. The benchmark "
        "refuses to run if no body matches.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)

    p.add_argument("--warmup-min", type=float, default=1.0, help="Walking warmup before perturbation (s)")
    p.add_argument("--warmup-max", type=float, default=2.0)
    p.add_argument("--max-pre-steps", type=int, default=150, help="Max control steps before giving up (3 s at 50 Hz)")
    p.add_argument("--post-steps", type=int, default=40, help="Post-trigger window (40 = 0.8 s at 50 Hz)")
    p.add_argument("--oracle-lead-steps", type=int, default=15, help="Oracle trigger lead (15 = 300 ms at 50 Hz)")

    # Perturbation magnitudes come from --preset; override individually to deviate.
    p.add_argument("--preset", choices=tuple(PERTURBATION_PRESETS), default="paper",
                   help="Perturbation strength profile. 'paper' (default) is the range in the "
                        "write-up and yields ~26%% falls at 64 envs; 'calibrated' is a stronger "
                        "stress profile (~62%%) for robustness sweeps.")
    p.add_argument("--push-min", type=float, default=None)
    p.add_argument("--push-max", type=float, default=None)
    p.add_argument("--slip-min", type=float, default=None)
    p.add_argument("--slip-max", type=float, default=None)
    p.add_argument("--slip-duration-min", type=float, default=None)
    p.add_argument("--slip-duration-max", type=float, default=None)
    p.add_argument("--trip-force-min", type=float, default=None)
    p.add_argument("--trip-force-max", type=float, default=None)
    p.add_argument("--trip-duration", type=float, default=None)
    p.add_argument(
        "--trip-mode",
        choices=TRIP_MODES,
        default="obstacle",
        help="'obstacle' (default) places a real box in front of the robot's motion at the "
        "trigger; 'force' applies a horizontal force to the swing foot (the previous "
        "synthetic trip, kept for reproducing older runs).",
    )
    p.add_argument(
        "--trip-height-min",
        type=float,
        default=0.05,
        help="Obstacle mode: lower bound of the per-trial top-face height above ground (m).",
    )
    p.add_argument(
        "--trip-height-max",
        type=float,
        default=0.15,
        help="Obstacle mode: upper bound of the per-trial top-face height above ground (m).",
    )
    p.add_argument(
        "--trip-lead-time",
        type=float,
        default=0.30,
        help="Obstacle mode: seconds of travel ahead of the base at which the box is placed.",
    )
    p.add_argument(
        "--trip-min-distance",
        type=float,
        default=0.25,
        help="Obstacle mode: minimum distance ahead of the base (m); floor of the clamp.",
    )
    p.add_argument(
        "--trip-max-distance",
        type=float,
        default=1.0,
        help="Obstacle mode: maximum distance ahead of the base (m); ceiling of the clamp.",
    )
    p.add_argument("--freeze-motor-off-p", type=float, default=None,
                   help="Probability a freeze scenario is motor-off rather than brake-lock.")
    p.add_argument("--freeze-two-joint-p", type=float, default=None,
                   help="Probability a freeze scenario disables two joints rather than one.")
    p.add_argument("--brake-kp", type=float, default=100.0)
    p.add_argument("--brake-kd", type=float, default=10.0)

    p.add_argument("--render-trial-id", type=int, default=-1, help="Render one trial instead of running the benchmark")
    p.add_argument("--render-baseline", choices=("nominal", "default", "damping"), default="nominal")
    p.add_argument("--render-camera", default="g1-chase", help="CAMERA_REGISTRY preset for the render camera")
    p.add_argument(
        "--render-backend",
        choices=("classic", "mjwarp"),
        default="classic",
        help="Backend for single-env rendering. mjwarp asserts device-side at num_envs=1.",
    )
    p.add_argument("--render-dir", default=None, help="Video output dir (default: <output-dir>/videos)")
    p.add_argument("--render-playback-rate", type=float, default=0.25)

    args = p.parse_args(argv)
    # Fill anything the caller did not set explicitly from the selected profile.
    for key, value in PERTURBATION_PRESETS[args.preset].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def main(argv: list[str] | None = None) -> None:
    init_eval_logging()
    args = parse_args(argv)
    bench = FallBenchmark(args)
    try:
        if args.render_trial_id >= 0:
            bench.render_trial()
        else:
            bench.run()
    finally:
        bench.close()


if __name__ == "__main__":
    main()
