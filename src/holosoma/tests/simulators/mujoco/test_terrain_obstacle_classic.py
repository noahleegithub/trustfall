"""Live CPU (ClassicBackend) test: the optional welded box obstacle on the terrain.

``TerrainTermCfg.obstacle`` is what the SafeFall benchmark's obstacle trip mode rides on:
the terrain builder adds a welded (jointless) box body, and the benchmark later relocates
it per environment through ``set_static_body_world_pose``. This test pins the parts that
can be checked without CUDA — the body/geom exists exactly when the config asks for it,
and relocating it moves both the body pose and the collision geom pose.

No GPU: the Warp analogue is the cluster diagnostic (``scripts/safefall/diag_trip_obstacle.py``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

pytestmark = pytest.mark.mujoco_classic

import mujoco  # noqa: E402

from holosoma.config_types.terrain import ObstacleCfg, with_obstacle  # noqa: E402
from holosoma.config_values.terrain import terrain_locomotion_plane  # noqa: E402
from tests.simulators.mujoco._build import build_classic_sim  # noqa: E402


def _body_id(model, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def _obstacle_geoms(model, body_id: int) -> list[int]:
    return [g for g in range(int(model.ngeom)) if int(model.geom_bodyid[g]) == body_id]


def test_obstacle_is_added_only_when_configured():
    plain = build_classic_sim()
    assert _body_id(plain.root_model, ObstacleCfg().name) < 0, "the box must be opt-in"

    cfg = ObstacleCfg()
    sim = build_classic_sim(terrain=with_obstacle(terrain_locomotion_plane, cfg))
    model = sim.root_model
    body_id = _body_id(model, cfg.name)
    assert body_id >= 0, "obstacle body missing from the compiled model"

    geoms = _obstacle_geoms(model, body_id)
    assert len(geoms) == 1
    geom = geoms[0]
    np.testing.assert_allclose(model.geom_size[geom], cfg.half_extents, atol=1e-6)
    assert int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_BOX)
    # Environment collision class: collides with the robot (class 1) only.
    assert int(model.geom_contype[geom]) == 2
    assert int(model.geom_conaffinity[geom]) == 1

    # Parked below the ground by default, so an un-moved box is inert.
    np.testing.assert_allclose(model.body_pos[body_id], cfg.pos, atol=1e-6)


def test_relocated_obstacle_moves_its_collision_geom():
    cfg = ObstacleCfg()
    sim = build_classic_sim(terrain=with_obstacle(terrain_locomotion_plane, cfg))
    body_id = _body_id(sim.root_model, cfg.name)
    geom = _obstacle_geoms(sim.root_model, body_id)[0]

    # 0.75 m ahead of the origin, yawed 90 degrees, top face at 0.10 m (box is 0.10 m tall,
    # centre therefore at 0.05).
    centre = np.array([0.75, 0.0, 0.10 - cfg.half_extents[2]])
    half = math.pi / 4.0
    quat_xyzw = torch.tensor([[[0.0, 0.0, math.sin(half), math.cos(half)]]], dtype=torch.float32)
    sim.backend.set_static_body_world_pose(
        [body_id],
        torch.tensor(centre, dtype=torch.float32).reshape(1, 1, 3),
        quat_xyzw,
        env_ids=torch.tensor([0]),
    )

    np.testing.assert_allclose(sim.root_model.body_pos[body_id], centre, atol=1e-5)
    # MuJoCo stores quaternions wxyz; the 90-degree yaw is [cos45, 0, 0, sin45].
    np.testing.assert_allclose(
        sim.root_model.body_quat[body_id], [math.cos(half), 0.0, 0.0, math.sin(half)], atol=1e-6
    )
    # The collision geom -- what the contact pipeline actually uses -- must follow the body.
    np.testing.assert_allclose(sim.backend.data.geom_xpos[geom], centre, atol=1e-5)
    # Top face at the requested height: centre + half height.
    assert math.isclose(float(sim.backend.data.geom_xpos[geom][2]) + cfg.half_extents[2], 0.10, abs_tol=1e-5)
