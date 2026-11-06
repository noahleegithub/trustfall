"""Unit test for the trip-obstacle glue in ``scripts/safefall/benchmark.py``.

Builds no simulator: ``_apply_factors`` is called with a stubbed sim/backend, so this
pins the parts of obstacle placement that a GPU run would otherwise be the only witness
to — that the box is written to the *trial's* env slot (not env 0), with the
``[n, 1, 3]`` / ``[n, 1, 4]`` shapes ``set_static_body_world_pose`` expects, and that the
resolved height/direction/distance land back on the scenario for the CSV.

Marked ``mujoco`` only because importing the benchmark module imports the MuJoCo package;
nothing here instantiates a sim, needs a checkpoint, or touches CUDA.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.mujoco

from scripts.safefall.benchmark import FallBenchmark, Scenario  # noqa: E402

N_ENVS = 6
OBSTACLE_BODY = 42
HALF_HEIGHT = 0.05


class _CaptureBackend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def set_static_body_world_pose(self, body_ids, positions, quats=None, env_ids=None):
        self.calls.append(
            {
                "body_ids": list(body_ids),
                "positions": positions.clone(),
                "quats": None if quats is None else quats.clone(),
                "env_ids": None if env_ids is None else env_ids.clone(),
            }
        )


class _StubSim:
    def __init__(self, n: int) -> None:
        self.robot_root_states = torch.zeros(n, 13)
        self.applied_forces = torch.zeros(n, 4, 6)
        self.contact_forces = torch.zeros(n, 2, 3)
        self._rigid_body_vel = torch.zeros(n, 2, 3)
        self.backend = _CaptureBackend()
        self.root_state_writes: list = []

    # The push/friction branches reach these; the trip-glue tests do not care what they do.
    def set_actor_root_state_tensor_robots(self, env_ids, states) -> None:
        self.root_state_writes.append((env_ids.clone(), states.clone()))

    def set_geom_friction(self, *args, **kwargs) -> None:
        pass

    def get_geom_friction(self, geom_ids, env_ids):
        return torch.zeros(len(env_ids), len(geom_ids))


class _StubMonitor:
    def clear_trial_start(self, env_ids) -> None:
        pass


def _bench() -> types.SimpleNamespace:
    args = types.SimpleNamespace(
        seed=0,
        trip_lead_time=0.30,
        trip_min_distance=0.25,
        trip_max_distance=1.0,
    )
    bench = types.SimpleNamespace(
        args=args,
        device="cpu",
        num_envs=N_ENVS,
        sim=_StubSim(N_ENVS),
        monitor=_StubMonitor(),
        obstacle_body_id=OBSTACLE_BODY,
        obstacle_half_height=HALF_HEIGHT,
        foot_holosoma_ids=[0, 1],
        foot_body_ids=[2, 3],
        freeze_mode=torch.zeros(N_ENVS, dtype=torch.long),
        freeze_mask=torch.zeros(N_ENVS, 4, dtype=torch.bool),
        freeze_q=torch.zeros(N_ENVS, 4),
    )
    # Bind the two methods ``_apply_factors`` calls on ``self``; the rest is state.
    for name in ("_set_applied", "_clear_applied"):
        setattr(bench, name, types.MethodType(getattr(FallBenchmark, name), bench))
    return bench


def _scenario(trial_id: int, height: float, command=(1.0, 0.0, 0.0)) -> Scenario:
    return Scenario(
        trial_id=trial_id,
        seed=trial_id,
        mechanisms=["trip"],
        command=command,
        t_pert=10,
        trip_height=height,
        trip_steps=0,
    )


def _apply(bench, scenarios, env_ids) -> None:
    FallBenchmark._apply_factors(
        bench,
        scenarios,
        torch.tensor(env_ids, dtype=torch.long),
        slip_end=torch.full((N_ENVS,), -1, dtype=torch.long),
        slip_mu=torch.zeros(N_ENVS),
        slip_baseline=torch.zeros(N_ENVS, 1),
        trip_end=torch.full((N_ENVS,), -1, dtype=torch.long),
        trip_force=torch.zeros(N_ENVS, 3),
        trip_body=torch.zeros(N_ENVS, dtype=torch.long),
    )


def test_box_is_written_to_the_trial_env_slot_with_backend_shapes():
    bench = _bench()
    # Two trip trials sharing a batch with a non-trip trial, placed in env slots 1 and 4.
    scenarios = [_scenario(2, 0.10), _scenario(6, 0.15)]
    bench.sim.robot_root_states[1, 0:2] = torch.tensor([0.0, 0.0])
    bench.sim.robot_root_states[1, 7:9] = torch.tensor([1.0, 0.0])  # 1 m/s +x -> 0.30 m ahead
    bench.sim.robot_root_states[4, 0:2] = torch.tensor([-2.0, 3.0])
    bench.sim.robot_root_states[4, 7:9] = torch.tensor([0.0, 0.0])  # standing -> min distance, command dir

    _apply(bench, scenarios, [1, 4])

    assert len(bench.sim.backend.calls) == 1, "one batched pose write for the whole scenario list"
    call = bench.sim.backend.calls[0]
    assert call["body_ids"] == [OBSTACLE_BODY]
    torch.testing.assert_close(call["env_ids"], torch.tensor([1, 4]))
    assert tuple(call["positions"].shape) == (2, 1, 3)
    assert tuple(call["quats"].shape) == (2, 1, 4)

    # Trial 0: velocity-driven, 0.30 m along +x, top face at 0.10 m.
    np.testing.assert_allclose(call["positions"][0, 0].numpy(), [0.30, 0.0, 0.05], atol=1e-5)
    # Trial 1: standing robot -> 0.25 m along the command (+x), top face at 0.15 m.
    np.testing.assert_allclose(call["positions"][1, 0].numpy(), [-2.0 + 0.25, 3.0, 0.10], atol=1e-5)
    np.testing.assert_allclose(call["quats"][0, 0].numpy(), [0.0, 0.0, 0.0, 1.0], atol=1e-6)

    # The resolved placement is reported back on the scenario for the CSV/JSONL.
    assert scenarios[0].trip_height == 0.10 and scenarios[1].trip_height == 0.15
    np.testing.assert_allclose([scenarios[0].trip_distance, scenarios[1].trip_distance], [0.30, 0.25], atol=1e-6)
    np.testing.assert_allclose([scenarios[0].trip_dir_x, scenarios[0].trip_dir_y], [1.0, 0.0], atol=1e-6)


def test_no_write_when_the_batch_has_no_obstacle_trip():
    bench = _bench()
    push_only = Scenario(trial_id=0, seed=0, mechanisms=["push"], command=(1.0, 0.0, 0.0), t_pert=10,
                         push_dv=(1.0, 0.5, 0.0))
    _apply(bench, [push_only], [0])
    assert bench.sim.backend.calls == [], "a non-trip trial must not move the box"


def test_force_mode_scenarios_never_touch_the_obstacle():
    bench = _bench()
    force_trip = Scenario(trial_id=2, seed=2, mechanisms=["trip"], command=(1.0, 0.0, 0.0), t_pert=10,
                          trip_force=150.0, trip_steps=3)
    _apply(bench, [force_trip], [0])
    assert bench.sim.backend.calls == [], "force-mode trip keeps the old foot-force path only"
