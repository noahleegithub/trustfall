"""Unit tests for the SafeFall trip-obstacle placement maths (no simulator, CPU only).

The maths lives in ``scripts/safefall/trip_placement.py`` as pure numpy so it can be
checked without MuJoCo or a GPU: these tests pin the direction fallback chain, the
distance clamp, the height convention (top face at the requested height) and the yaw
quaternion.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

# Load the module directly, like test_impact_metrics.py: importing the scripts package is
# not needed and must not drag in the benchmark (which imports holosoma + torch).
_MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts/safefall/trip_placement.py"
_spec = importlib.util.spec_from_file_location("_trip_placement_test", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_trip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_trip)

HALF_HEIGHT = 0.05  # ObstacleCfg.half_extents[2]: a 0.10 m tall box


def _quat_rotated_xaxis(quat_xyzw) -> np.ndarray:
    """Rotate the local +x axis by a yaw-only xyzw quaternion (what the box face implies)."""
    qx, qy, qz, qw = quat_xyzw
    assert qx == 0.0 and qy == 0.0, "box orientation must be yaw-only"
    # Yaw from the quaternion, then the x axis it maps to.
    yaw = 2.0 * math.atan2(qz, qw)
    return np.array([math.cos(yaw), math.sin(yaw)])


# ---------------------------------------------------------------- direction fallback


def test_direction_uses_base_velocity_when_moving():
    direction = _trip.travel_direction([[0.0, 1.0]], command_xy=[[1.0, 0.0]])
    np.testing.assert_allclose(direction, [[0.0, 1.0]], atol=1e-12)


def test_direction_is_normalized_for_any_speed():
    direction = _trip.travel_direction([[3.0, 4.0]])  # 5 m/s at 3-4-5
    np.testing.assert_allclose(direction, [[0.6, 0.8]], atol=1e-12)
    assert direction.shape == (1, 2)


def test_stationary_robot_falls_back_to_command_direction():
    direction = _trip.travel_direction([[0.0, 0.0]], command_xy=[[0.0, -0.8]])
    np.testing.assert_allclose(direction, [[0.0, -1.0]], atol=1e-12)


def test_slow_robot_below_threshold_falls_back_to_command():
    # 0.10 m/s < 0.15 m/s threshold: measured velocity is noise, use the command.
    direction = _trip.travel_direction([[0.10, 0.0]], command_xy=[[0.0, 0.5]])
    np.testing.assert_allclose(direction, [[0.0, 1.0]], atol=1e-12)


def test_velocity_at_threshold_is_used():
    direction = _trip.travel_direction([[0.15, 0.0]], command_xy=[[0.0, 1.0]])
    np.testing.assert_allclose(direction, [[1.0, 0.0]], atol=1e-12)


def test_stationary_with_zero_command_falls_back_to_plus_x():
    direction = _trip.travel_direction([[0.0, 0.0]], command_xy=[[0.0, 0.0]])
    np.testing.assert_allclose(direction, [[1.0, 0.0]], atol=1e-12)


def test_stationary_without_command_falls_back_to_plus_x():
    direction = _trip.travel_direction([[0.0, 0.0]])
    np.testing.assert_allclose(direction, [[1.0, 0.0]], atol=1e-12)


def test_direction_is_per_row():
    direction = _trip.travel_direction([[1.0, 0.0], [0.0, 0.0]], command_xy=[[0.0, 1.0], [0.0, -1.0]])
    np.testing.assert_allclose(direction, [[1.0, 0.0], [0.0, -1.0]], atol=1e-12)


# ---------------------------------------------------------------- distance clamp


def test_distance_scales_with_speed_inside_the_band():
    # 1.0 m/s * 0.30 s = 0.30 m, inside [0.25, 1.0].
    np.testing.assert_allclose(_trip.travel_distance([1.0]), [0.30], atol=1e-12)


def test_distance_clamped_at_minimum_when_slow():
    np.testing.assert_allclose(_trip.travel_distance([0.0]), [_trip.MIN_DISTANCE_M], atol=1e-12)


def test_distance_clamped_at_maximum_when_fast():
    # 10 m/s * 0.30 s = 3.0 m -> clamped to 1.0 m.
    np.testing.assert_allclose(_trip.travel_distance([10.0]), [_trip.MAX_DISTANCE_M], atol=1e-12)


def test_distance_respects_custom_band():
    got = _trip.travel_distance([0.0, 2.0], lead_time=0.5, min_distance=0.1, max_distance=0.6)
    np.testing.assert_allclose(got, [0.1, 0.6], atol=1e-12)


def test_reversed_distance_band_is_rejected():
    try:
        _trip.travel_distance([1.0], min_distance=2.0, max_distance=1.0)
    except ValueError:
        pass
    else:  # pragma: no cover - the guard is the point of the test
        raise AssertionError("min_distance > max_distance should raise")


# ---------------------------------------------------------------- full plan


def test_plan_puts_the_top_face_at_the_requested_height():
    for height in (0.05, 0.10, 0.15):
        plan = _trip.plan_trip_obstacle(
            base_xy=[[0.0, 0.0]],
            velocity_xy=[[1.0, 0.0]],
            command_xy=[[1.0, 0.0]],
            trip_height=height,
            half_height=HALF_HEIGHT,
        )
        # Centre sits half a box below the requested top face...
        assert math.isclose(plan.center[0, 2], height - HALF_HEIGHT, abs_tol=1e-12)
        # ...so the top face lands exactly at the requested height above the ground (z=0).
        assert math.isclose(plan.center[0, 2] + HALF_HEIGHT, height, abs_tol=1e-12)


def test_plan_places_the_box_ahead_along_the_direction_of_travel():
    plan = _trip.plan_trip_obstacle(
        base_xy=[[1.0, 2.0]],
        velocity_xy=[[0.0, 1.5]],  # 1.5 m/s +y -> 0.45 m ahead
        command_xy=[[1.0, 0.0]],
        trip_height=0.10,
        half_height=HALF_HEIGHT,
    )
    assert math.isclose(plan.distance[0], 1.5 * _trip.LEAD_TIME_S, abs_tol=1e-12)
    np.testing.assert_allclose(plan.center[0, :2], [1.0, 2.0 + 1.5 * _trip.LEAD_TIME_S], atol=1e-12)


def test_plan_of_a_stationary_robot_uses_min_distance_along_the_command():
    plan = _trip.plan_trip_obstacle(
        base_xy=[[0.0, 0.0]],
        velocity_xy=[[0.0, 0.0]],
        command_xy=[[-1.0, 0.0]],
        trip_height=0.15,
        half_height=HALF_HEIGHT,
    )
    np.testing.assert_allclose(plan.center[0, :2], [-_trip.MIN_DISTANCE_M, 0.0], atol=1e-12)
    np.testing.assert_allclose(plan.direction[0], [-1.0, 0.0], atol=1e-12)


def test_plan_yaw_quaternion_faces_the_box_perpendicular_to_travel():
    plan = _trip.plan_trip_obstacle(
        base_xy=[[0.0, 0.0]],
        velocity_xy=[[0.0, 2.0]],  # travel +y
        command_xy=[[0.0, 1.0]],
        trip_height=0.10,
        half_height=HALF_HEIGHT,
    )
    np.testing.assert_allclose(plan.quat[0], [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)], atol=1e-12)
    np.testing.assert_allclose(_quat_rotated_xaxis(plan.quat[0]), plan.direction[0], atol=1e-9)


def test_plan_quaternion_yaw_matches_direction_for_an_oblique_heading():
    plan = _trip.plan_trip_obstacle(
        base_xy=[[0.0, 0.0]],
        velocity_xy=[[2.0, 0.0]],
        command_xy=[[1.0, 0.0]],
        trip_height=0.05,
        half_height=HALF_HEIGHT,
    )
    np.testing.assert_allclose(_quat_rotated_xaxis(plan.quat[0]), [1.0, 0.0], atol=1e-9)
    assert math.isclose(np.linalg.norm(plan.quat[0]), 1.0, abs_tol=1e-12)


def test_plan_is_batched_over_environments():
    plan = _trip.plan_trip_obstacle(
        base_xy=np.array([[0.0, 0.0], [4.0, -1.0]]),
        velocity_xy=np.array([[1.0, 0.0], [0.0, 0.0]]),
        command_xy=np.array([[1.0, 0.0], [0.0, 1.0]]),
        trip_height=np.array([0.08, 0.12]),
        half_height=HALF_HEIGHT,
    )
    assert plan.center.shape == (2, 3) and plan.quat.shape == (2, 4) and plan.direction.shape == (2, 2)
    np.testing.assert_allclose(plan.center[0, :2], [0.30, 0.0], atol=1e-12)  # 1 m/s * 0.3 s
    np.testing.assert_allclose(plan.center[1, :2], [4.0, -1.0 + _trip.MIN_DISTANCE_M], atol=1e-12)
    np.testing.assert_allclose(plan.center[:, 2] + HALF_HEIGHT, [0.08, 0.12], atol=1e-12)


def test_negative_height_is_rejected():
    try:
        _trip.plan_trip_obstacle(
            base_xy=[[0.0, 0.0]],
            velocity_xy=[[1.0, 0.0]],
            command_xy=[[1.0, 0.0]],
            trip_height=-0.01,
            half_height=HALF_HEIGHT,
        )
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a negative top-face height should raise")
