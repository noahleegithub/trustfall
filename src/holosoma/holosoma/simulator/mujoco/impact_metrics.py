"""Impact metrics for SafeFall-style fall evaluation.

This module implements the seven metrics used by the SafeFall paper (Meng et al.,
arXiv:2511.18509) and a small per-substep recorder that turns simulator signals into
them:

* ``max_joint_torque`` / ``max_torque_ratio`` -- peak ``|qfrc_constraint|`` over the
  robot's revolute joints (constraint/external torque, excluding motor output).
* ``max_joint_reaction`` -- peak force magnitude of ``cfrc_int`` at each joint's child
  body (the wrench transmitted through the joint).
* ``max_contact_force`` -- peak per-link ground contact force (robot link vs. a
  non-robot body), adjacent-link self contacts excluded.
* ``peak_force_{head,torso,hand}`` / ``cum_impulse_*`` -- per-part severity, attributed
  by GEOM (the G1 head capsule is welded to ``torso_link``, so it is invisible to body
  masks). These count ground / non-robot contacts only: a hand striking the head during
  a fall is not included.
* ``max_impulse`` / ``cumulative_impulse`` -- max per-physics-step and summed ground
  reaction impulse over the evaluated window.
* ``illegal_contact`` -- whether a protected link (head / hands) contacted the ground
  or a non-adjacent link.
* ``max_limit_violations`` -- max number of joints outside their position limits at
  one physics step.

The monitor samples every physics substep (via ``Phase.POST_STEP``), keeps one
control-step summary in a small ring buffer, and can return trial peaks for a
requested control-step window.  It deliberately keeps all accumulation on the
simulator device and only copies to the host when results are requested.

Notes
-----
* ``cfrc_int`` / ``cfrc_ext`` are ``[torque(3), force(3)]`` spatial vectors (the
  MuJoCo ``c-frame`` convention).  Only the force half is used here.
* The per-contact force returned by the backends is the force acting on the *second*
  geom's body.  The monitor flips the sign for contacts where the robot body is
  ``geom1``.
* Ground is defined as "the other body is not a robot body".  This covers the world
  body, terrain, and scene objects; the benchmark uses flat ground, so everything
  non-robot is ground.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from holosoma.simulator.base_simulator.hooks import Phase


@dataclass
class ImpactTrialMetrics:
    """One trial's peak metrics, as CPU tensors of shape ``[num_envs]``."""

    max_joint_torque: torch.Tensor
    max_torque_ratio: torch.Tensor
    max_joint_reaction: torch.Tensor
    max_contact_force: torch.Tensor
    max_impulse: torch.Tensor
    cumulative_impulse: torch.Tensor
    illegal_contact: torch.Tensor
    torso_contact: torch.Tensor
    peak_force_head: torch.Tensor
    peak_force_torso: torch.Tensor
    peak_force_hand: torch.Tensor
    cum_impulse_head: torch.Tensor
    cum_impulse_torso: torch.Tensor
    cum_impulse_hand: torch.Tensor
    max_limit_violations: torch.Tensor


class ImpactMonitor:
    """Per-substep SafeFall metric recorder.

    Parameters
    ----------
    env:
        The environment (``BaseTask``) owning the simulator.  Used for DOF limits and
        torque limits.
    simulator:
        The active MuJoCo simulator (``holosoma.simulator.mujoco.mujoco.MuJoCo``).
    window_steps:
        Ring-buffer length in control steps.  Must be greater than the longest window
        passed to :meth:`results` (the benchmark uses 40).
    protected_pattern, foot_pattern:
        Case-insensitive regex-ish substring patterns for protected and foot body
        names.  ``head``/``hand`` and ``foot``/``ankle`` are the defaults.
    """

    def __init__(
        self,
        env,
        simulator,
        *,
        window_steps: int = 48,
        protected_patterns: Sequence[str] = ("head", "hand"),
        foot_patterns: Sequence[str] = ("foot", "ankle"),
        severe_patterns: Sequence[str] = ("torso",),
    ) -> None:
        if window_steps < 1:
            raise ValueError(f"window_steps must be >= 1, got {window_steps}")
        self.env = env
        self.simulator = simulator
        self.backend = simulator.backend
        self.device = torch.device(simulator.sim_device)
        self.num_envs = int(simulator.num_envs)
        self.nbody = int(simulator.root_model.nbody)
        self.ndof = int(simulator.num_dof)
        self.dt = float(simulator.sim_dt)
        self.window_steps = int(window_steps)

        robot_body_ids = torch.as_tensor(list(simulator.body_ids), dtype=torch.long, device=self.device)
        self.robot_body_ids = robot_body_ids
        self.is_robot = torch.zeros(self.nbody, dtype=torch.bool, device=self.device)
        self.is_robot[robot_body_ids] = True

        lower_names = [str(n).lower() for n in simulator.body_names]
        self.foot_mask = self._body_mask(lower_names, foot_patterns)
        self.protected_mask = self._body_mask(lower_names, protected_patterns)
        # Torso contact is reported separately from illegal (head/hand) contact: on a
        # headless humanoid every fall ends on the torso, so folding it into
        # illegal_contact would saturate that metric at 100% and destroy its ranking.
        self.severe_mask = self._body_mask(lower_names, severe_patterns)

        # ---- part attribution (head / torso / hand) by GEOM, not body.
        # On the G1 the head collision capsule is `<geom name="head_collision">`
        # welded to torso_link, so body-level attribution merges head strikes into
        # torso strikes. Geom ids arrive with every contact, so classify per contact.
        self.PARTS: tuple[str, ...] = ("head", "torso", "hand")
        geom_names = list(getattr(simulator, "geom_names", []) or [])
        ngeom_model = int(simulator.root_model.ngeom)
        if len(geom_names) != ngeom_model:
            raise RuntimeError(
                f"simulator.geom_names has {len(geom_names)} entries but the model has "
                f"{ngeom_model} geoms; per-part impact attribution would report zeros "
                "for every part instead of failing. Refusing to build the monitor."
            )
        body_of_geom = np.asarray(simulator.root_model.geom_bodyid, dtype=np.int64)
        body_name_by_id = {int(b): n for b, n in zip(list(simulator.body_ids), lower_names)}
        self.geom_part = torch.full((max(len(geom_names), 1),), -1, dtype=torch.long, device=self.device)
        for g, gname in enumerate(geom_names):
            ln = str(gname).lower()
            bn = body_name_by_id.get(int(body_of_geom[g]), "") if g < len(body_of_geom) else ""
            if "head" in ln or "head" in bn:
                self.geom_part[g] = 0
            elif "torso" in ln or "torso" in bn:
                self.geom_part[g] = 1
            elif "wrist" in ln or "hand" in ln or "wrist" in bn or "hand" in bn:
                self.geom_part[g] = 2
        if int((self.geom_part >= 0).sum()) == 0:
            raise RuntimeError(
                "No geom matched any impact part (head/torso/hand); per-part severity "
                "would be identically zero. Refusing to build the monitor."
            )

        # Raw body ids (0 == world).  ``body_parentid`` is static.
        parent = np.asarray(simulator.root_model.body_parentid, dtype=np.int64)
        self.parent = torch.as_tensor(parent, dtype=torch.long, device=self.device)

        # Robot DOF -> raw child body id (for cfrc_int).  ``dof_qvel_addrs`` are the
        # model DOF addresses, so resolve through dof_jntid -> jnt_bodyid without any
        # name lookups (and without importing mujoco here).
        dof_addrs_np = np.asarray(list(simulator.dof_qvel_addrs), dtype=np.int64)
        dof_jntid = np.asarray(simulator.root_model.dof_jntid, dtype=np.int64)[dof_addrs_np]
        dof_body_ids = np.asarray(simulator.root_model.jnt_bodyid, dtype=np.int64)[dof_jntid]
        self.dof_body_ids = torch.as_tensor(dof_body_ids, dtype=torch.long, device=self.device)
        self.dof_idx = torch.as_tensor(list(simulator.dof_qvel_addrs), dtype=torch.long, device=self.device)

        # Static limits.  Prefer the hard limits for "joint position limit" violations;
        # ``dof_pos_limits`` is inflated by the soft-limit factor used for termination.
        hard_limits = getattr(env, "hard_dof_pos_limits", env.dof_pos_limits)
        self.q_lower = torch.as_tensor(hard_limits[:, 0], dtype=torch.float32, device=self.device)
        self.q_upper = torch.as_tensor(hard_limits[:, 1], dtype=torch.float32, device=self.device)
        self.tau_max = torch.as_tensor(env.torque_limits, dtype=torch.float32, device=self.device).clamp_min(1e-6)

        self.current_step = 0
        k = self.window_steps
        n = self.num_envs
        self._ring = {
            "joint_torque": torch.zeros(k, n, device=self.device),
            "torque_ratio": torch.zeros(k, n, device=self.device),
            "joint_reaction": torch.zeros(k, n, device=self.device),
            "contact_force": torch.zeros(k, n, device=self.device),
            "impulse": torch.zeros(k, n, device=self.device),
            "limit_violations": torch.zeros(k, n, device=self.device),
            "illegal": torch.zeros(k, n, device=self.device),
            "severe": torch.zeros(k, n, device=self.device),
            # per-part severity: peak force and accumulated impulse per control step
            **{f"force_{p}": torch.zeros(k, n, device=self.device) for p in ("head", "torso", "hand")},
            **{f"imp_{p}": torch.zeros(k, n, device=self.device) for p in ("head", "torso", "hand")},
        }
        self._step = {key: torch.zeros(n, device=self.device) for key in self._ring}
        # Reused scratch for per-part force accumulation (avoids alloc + sync per substep).
        self._part_acc = torch.zeros(n, 3, device=self.device)
        self.first_contact_step = torch.full((n,), -1, dtype=torch.long, device=self.device)

        # Hook registration.  FRAME_BEGIN / FRAME_END bracket one control step; POST_STEP
        # fires after every physics substep.
        simulator.hooks.add(Phase.FRAME_BEGIN, self._on_frame_begin, name="impact.frame_begin")
        simulator.hooks.add(Phase.POST_STEP, self._on_post_step, name="impact.post_step")
        simulator.hooks.add(Phase.FRAME_END, self._on_frame_end, name="impact.frame_end")

    # ------------------------------------------------------------------ helpers

    def _body_mask(self, lower_names: list[str], patterns: Sequence[str]) -> torch.Tensor:
        mask = torch.zeros(self.nbody, dtype=torch.bool, device=self.device)
        for body_id, name in zip(self.simulator.body_ids, lower_names, strict=False):
            if any(p.lower() in name for p in patterns):
                mask[int(body_id)] = True
        return mask

    def _reset_step_metrics(self) -> None:
        for value in self._step.values():
            value.zero_()

    def clear_trial_start(self, env_ids: torch.Tensor | None = None) -> None:
        """Forget pre-perturbation contacts (call when a scenario's perturbation fires)."""
        if env_ids is None:
            self.first_contact_step.fill_(-1)
        elif len(env_ids) > 0:
            self.first_contact_step[env_ids] = -1

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear accumulated metrics for a set of environments (and first-contact state)."""
        if env_ids is None:
            self.current_step = 0
            for ring in self._ring.values():
                ring.zero_()
            self._reset_step_metrics()
            self.first_contact_step.fill_(-1)
            return
        if len(env_ids) == 0:
            return
        for ring in self._ring.values():
            ring[:, env_ids] = 0.0
        for value in self._step.values():
            value[env_ids] = 0.0
        self.first_contact_step[env_ids] = -1

    # ------------------------------------------------------------------ hooks

    def _on_frame_begin(self) -> None:
        self.current_step += 1
        self._reset_step_metrics()

    def _on_frame_end(self) -> None:
        idx = self.current_step % self.window_steps
        for key, ring in self._ring.items():
            ring[idx] = self._step[key]

    def _on_post_step(self) -> None:
        self.update()

    # ------------------------------------------------------------------ sampling

    def update(self) -> None:
        """Sample the simulator once (after a physics substep) and fold into this step's maxima."""
        sim = self.simulator
        backend = self.backend

        qfrc_constraint = backend.get_constraint_forces()  # [N, nv]
        cfrc_int = backend.get_internal_wrenches()  # [N, nbody, 6] torque:force
        (
            contact_force,
            body1,
            body2,
            worldid,
            valid,
            geom1_raw,
            geom2_raw,
        ) = backend.get_contact_forces_world()

        n = self.num_envs
        # ---- joint torque / torque ratio
        tau = qfrc_constraint[:, self.dof_idx].abs()
        self._step["joint_torque"] = torch.maximum(self._step["joint_torque"], tau.max(dim=1).values)
        ratio = (tau / self.tau_max.unsqueeze(0)).max(dim=1).values
        self._step["torque_ratio"] = torch.maximum(self._step["torque_ratio"], ratio)

        # ---- joint reaction force (cfrc_int force half; [3:6] is force)
        reaction = torch.linalg.vector_norm(cfrc_int[:, self.dof_body_ids, 3:6], dim=-1)
        self._step["joint_reaction"] = torch.maximum(self._step["joint_reaction"], reaction.max(dim=1).values)

        # ---- contacts
        # Contact force is world-frame force on geom2's body.  Flip for geom1's body.
        if contact_force.numel() == 0:
            ground_force = torch.zeros(n, self.nbody, 3, device=self.device)
            nonfoot_contact = torch.zeros(n, dtype=torch.bool, device=self.device)
            illegal_contact = torch.zeros(n, dtype=torch.bool, device=self.device)
            severe_contact = torch.zeros(n, dtype=torch.bool, device=self.device)
        else:
            body1 = body1.to(device=self.device, dtype=torch.long)
            body2 = body2.to(device=self.device, dtype=torch.long)
            worldid = worldid.to(device=self.device, dtype=torch.long)
            valid = valid.to(device=self.device, dtype=torch.bool)
            contact_force = contact_force.to(device=self.device, dtype=torch.float32)
            # Padded Warp contact slots can contain arbitrary ids; never index with them.
            body1 = torch.where(valid, body1, torch.zeros_like(body1)).clamp_(0, self.nbody - 1)
            body2 = torch.where(valid, body2, torch.zeros_like(body2)).clamp_(0, self.nbody - 1)
            worldid = torch.where(valid, worldid, torch.zeros_like(worldid)).clamp_(0, n - 1)

            robot1 = self.is_robot[body1]
            robot2 = self.is_robot[body2]
            external = valid & (robot1 ^ robot2)
            robot_self = valid & robot1 & robot2

            # Ground force per robot body.  For an external contact exactly one side is
            # the robot; use the force acting on that body.
            robot_body = torch.where(robot1, body1, body2)
            sign = torch.where(robot1, -1.0, 1.0)
            f_robot = contact_force * sign.unsqueeze(-1)
            flat_body = torch.where(external, robot_body, torch.zeros_like(robot_body))
            flat_world = worldid
            flat_index = flat_world * self.nbody + flat_body
            flat_force = torch.where(external.unsqueeze(-1), f_robot, torch.zeros_like(f_robot))
            ground_flat = torch.zeros(n * self.nbody, 3, device=self.device)
            ground_flat.index_add_(0, flat_index, flat_force)
            ground_force = ground_flat.view(n, self.nbody, 3)

            # Non-foot ground contact (used for T / impact detection).
            nonfoot = external & ~self.foot_mask[robot_body]
            nonfoot_contact = _scatter_any(nonfoot, worldid, n)

            # ---- part attribution (head / torso / hand) by GEOM. Computed before the
            # boolean flags so they agree with the severity columns: on the G1 the head
            # is a capsule welded to torso_link, which no body mask can represent.
            g1 = geom1_raw.to(device=self.device, dtype=torch.long).clamp(0, self.geom_part.shape[0] - 1)
            g2 = geom2_raw.to(device=self.device, dtype=torch.long).clamp(0, self.geom_part.shape[0] - 1)
            part_of_contact = torch.where(robot1, self.geom_part[g1], self.geom_part[g2])
            head_part = self.PARTS.index("head")
            torso_part = self.PARTS.index("torso")
            head_contact = external & (part_of_contact == head_part)

            # Illegal: protected link vs ground or a non-adjacent robot link -- plus any
            # head-capsule ground strike, which the body mask cannot see.
            protected = self.protected_mask[body1] | self.protected_mask[body2]
            adjacent = (self.parent[body1] == body2) | (self.parent[body2] == body1)
            illegal = valid & ((protected & (external | (robot_self & ~adjacent))) | head_contact)
            illegal_contact = _scatter_any(illegal, worldid, n)

            # Torso ground contact, classified by geom so head strikes are NOT counted
            # as torso strikes (they share torso_link's body).
            severe = external & (part_of_contact == torso_part)
            severe_contact = _scatter_any(severe, worldid, n)

            # ---- per-part impact severity. Vector-sum each part's contacts per substep
            # so simultaneous contacts are not individually counted as the peak, then
            # track the maximum force and the accumulated impulse. The scratch buffer is
            # reused and zeroed in place: no per-substep allocation and no device sync.
            acc = self._part_acc
            for pid, pname in enumerate(self.PARTS):
                selected = external & (part_of_contact == pid)
                acc.zero_()
                acc.index_add_(0, worldid[selected], f_robot[selected])
                magnitude = torch.linalg.vector_norm(acc, dim=-1)
                self._step[f"force_{pname}"] = torch.maximum(self._step[f"force_{pname}"], magnitude)
                self._step[f"imp_{pname}"] = self._step[f"imp_{pname}"] + magnitude * self.dt

        # Per-link ground force (raw peak; the reward's [F - mg]+ form is built later).
        link_force = torch.linalg.vector_norm(ground_force[:, self.robot_body_ids, :], dim=-1)
        self._step["contact_force"] = torch.maximum(self._step["contact_force"], link_force.max(dim=1).values)

        total_force = ground_force.sum(dim=1)
        impulse = torch.linalg.vector_norm(total_force, dim=-1) * self.dt
        self._step["impulse"] = torch.maximum(self._step["impulse"], impulse)

        self._step["illegal"] = torch.maximum(
            self._step["illegal"], illegal_contact.to(dtype=self._step["illegal"].dtype)
        )

        self._step["severe"] = torch.maximum(
            self._step["severe"], severe_contact.to(dtype=self._step["severe"].dtype)
        )

        # ---- joint limit violations
        q = sim.dof_pos[:]
        violations = ((q < self.q_lower - 1e-4) | (q > self.q_upper + 1e-4)).sum(dim=1).to(torch.float32)
        self._step["limit_violations"] = torch.maximum(self._step["limit_violations"], violations)

        # ---- first non-foot ground contact of the trial (for T)
        newly = nonfoot_contact & (self.first_contact_step < 0)
        if newly.any():
            self.first_contact_step[newly] = self.current_step

    # ------------------------------------------------------------------ results

    def results(
        self,
        env_ids: torch.Tensor,
        start_steps: torch.Tensor,
        end_steps: torch.Tensor,
    ) -> ImpactTrialMetrics:
        """Return peak metrics over ``[start, end]`` (inclusive) for selected envs.

        ``start_steps`` / ``end_steps`` are control-step indices (1-based, matching
        :attr:`current_step`).  The window must fit in the ring buffer.
        """
        if len(env_ids) == 0:
            empty = torch.zeros(0, device=self.device)
            return ImpactTrialMetrics(
                max_joint_torque=empty,
                max_torque_ratio=empty.clone(),
                max_joint_reaction=empty.clone(),
                max_contact_force=empty.clone(),
                max_impulse=empty.clone(),
                cumulative_impulse=empty.clone(),
                illegal_contact=empty.clone(),
                torso_contact=empty.clone(),
                peak_force_head=empty.clone(),
                peak_force_torso=empty.clone(),
                peak_force_hand=empty.clone(),
                cum_impulse_head=empty.clone(),
                cum_impulse_torso=empty.clone(),
                cum_impulse_hand=empty.clone(),
                max_limit_violations=empty.clone(),
            )
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        start = start_steps.to(device=self.device, dtype=torch.long)
        end = end_steps.to(device=self.device, dtype=torch.long)
        lengths = (end - start + 1).clamp_min(1)
        if bool((lengths > self.window_steps).any()):
            raise ValueError(f"Requested window longer than the impact ring buffer ({self.window_steps} steps).")
        arange = torch.arange(int(lengths.max()), device=self.device)
        idx = (start.unsqueeze(1) + arange.unsqueeze(0)) % self.window_steps  # [n_sel, max_len]
        torch.arange(len(env_ids), device=self.device).unsqueeze(1)
        mask = arange.unsqueeze(0) < lengths.unsqueeze(1)

        def gather(key: str) -> torch.Tensor:
            ring = self._ring[key][idx, env_ids.unsqueeze(1)]  # [n_sel, max_len]
            # Zero-out padding so max/sum are well-defined for shorter windows.
            return ring * mask

        torque = gather("joint_torque").max(dim=1).values
        ratio = gather("torque_ratio").max(dim=1).values
        reaction = gather("joint_reaction").max(dim=1).values
        contact = gather("contact_force").max(dim=1).values
        impulse = gather("impulse")
        max_impulse = impulse.max(dim=1).values
        cumulative = impulse.sum(dim=1)
        illegal = gather("illegal").max(dim=1).values > 0.5
        torso = gather("severe").max(dim=1).values > 0.5
        peak_force = {p: gather(f"force_{p}").max(dim=1).values for p in self.PARTS}
        cum_impulse = {p: gather(f"imp_{p}").sum(dim=1) for p in self.PARTS}
        limits = gather("limit_violations").max(dim=1).values

        return ImpactTrialMetrics(
            max_joint_torque=torque,
            max_torque_ratio=ratio,
            max_joint_reaction=reaction,
            max_contact_force=contact,
            max_impulse=max_impulse,
            cumulative_impulse=cumulative,
            illegal_contact=illegal,
            torso_contact=torso,
            peak_force_head=peak_force["head"],
            peak_force_torso=peak_force["torso"],
            peak_force_hand=peak_force["hand"],
            cum_impulse_head=cum_impulse["head"],
            cum_impulse_torso=cum_impulse["torso"],
            cum_impulse_hand=cum_impulse["hand"],
            max_limit_violations=limits,
        )


def _scatter_any(flags: torch.Tensor, worldid: torch.Tensor, num_envs: int) -> torch.Tensor:
    """OR contact flags into a per-environment boolean vector."""
    out = torch.zeros(num_envs, dtype=torch.float32, device=flags.device)
    if flags.numel() == 0:
        return out > 0.5
    out.index_add_(0, worldid.to(device=flags.device, dtype=torch.long), flags.to(torch.float32))
    return out > 0.5
