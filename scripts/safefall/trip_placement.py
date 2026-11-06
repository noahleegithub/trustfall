"""Placement maths for the SafeFall trip hazard (no MuJoCo, no GPU).

The ``trip`` perturbation used to be a synthetic force on the swing foot. In obstacle
mode it is a real box that is moved in front of the robot at the moment the perturbation
fires, so the swing foot physically catches on it. Everything that decides *where* the
box goes lives here as plain numpy so it can be unit-tested without a simulator.

Conventions
-----------
* Every function takes and returns plain numpy arrays with a leading ``(N, ...)``
  per-environment axis. A single 2-vector is accepted as ``(1, 2)`` after reshaping.
* ``quat`` is xyzw (holosoma's convention); the box only ever carries yaw.
* Positions are world coordinates. The terrain term broadcasts one spawn origin to every
  env, so a robot's world ``xy`` is also the position the box is placed relative to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEAD_TIME_S = 0.30
"""Seconds of travel ahead of the robot at which the box is placed."""

MIN_DISTANCE_M = 0.25
MAX_DISTANCE_M = 1.0

MIN_SPEED_MPS = 0.15
"""Below this base speed the measured velocity direction is noise (or the robot is
standing); fall back to the commanded direction, then to +x."""

_EPS = 1e-6


@dataclass(frozen=True)
class TripPlan:
    """Where one box goes, per environment (all arrays carry a leading env axis)."""

    direction: np.ndarray
    """``(N, 2)`` unit xy heading of travel."""
    distance: np.ndarray
    """``(N,)`` metres ahead of the base along ``direction``."""
    center: np.ndarray
    """``(N, 3)`` box centre in world coordinates."""
    quat: np.ndarray
    """``(N, 4)`` xyzw yaw-only orientation."""


def _rows_xy(x) -> np.ndarray:
    """Coerce ``(2,)`` or ``(N, 2)`` input to a float64 ``(N, 2)`` array."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError(f"expected an xy vector, got a scalar: {x!r}")
    return arr.reshape(-1, 2)


def travel_direction(
    velocity_xy,
    command_xy=None,
    *,
    min_speed: float = MIN_SPEED_MPS,
    eps: float = _EPS,
) -> np.ndarray:
    """Unit xy direction of travel, with the documented fallback chain.

    ``norm(velocity) >= min_speed`` gives the measured heading. Otherwise the commanded
    direction is used when it is non-degenerate, and ``+x`` when it is not.
    """
    vel = _rows_xy(velocity_xy)
    out = np.zeros_like(vel)
    speed = np.linalg.norm(vel, axis=1)
    moving = speed >= min_speed
    out[moving] = vel[moving] / speed[moving, None]

    falling_back = ~moving
    if command_xy is not None:
        cmd = _rows_xy(command_xy)
        if cmd.shape[0] != vel.shape[0]:
            raise ValueError(f"command has {cmd.shape[0]} rows, velocity has {vel.shape[0]}")
        cmd_speed = np.linalg.norm(cmd, axis=1)
        use_cmd = falling_back & (cmd_speed > eps)
        out[use_cmd] = cmd[use_cmd] / cmd_speed[use_cmd, None]
        falling_back = falling_back & ~use_cmd

    out[falling_back] = (1.0, 0.0)
    return out


def travel_distance(
    speed,
    *,
    lead_time: float = LEAD_TIME_S,
    min_distance: float = MIN_DISTANCE_M,
    max_distance: float = MAX_DISTANCE_M,
) -> np.ndarray:
    """Distance ahead: ``clamp(speed * lead_time, min_distance, max_distance)``."""
    if min_distance > max_distance:
        raise ValueError(f"min_distance {min_distance} > max_distance {max_distance}")
    s = np.atleast_1d(np.asarray(speed, dtype=np.float64))
    return np.clip(s * lead_time, min_distance, max_distance)


def yaw_quaternion_xyzw(yaw) -> np.ndarray:
    """``(N, 4)`` xyzw quaternion rotating by ``yaw`` about +z."""
    y = np.atleast_1d(np.asarray(yaw, dtype=np.float64))
    out = np.zeros((y.shape[0], 4), dtype=np.float64)
    half = 0.5 * y
    out[:, 2] = np.sin(half)
    out[:, 3] = np.cos(half)
    return out


def plan_trip_obstacle(
    base_xy,
    velocity_xy,
    command_xy,
    trip_height,
    *,
    half_height: float,
    lead_time: float = LEAD_TIME_S,
    min_distance: float = MIN_DISTANCE_M,
    max_distance: float = MAX_DISTANCE_M,
    min_speed: float = MIN_SPEED_MPS,
) -> TripPlan:
    """Place a box ``lead_time`` seconds of travel ahead of each robot's base.

    ``trip_height`` is the requested TOP-FACE height above the ground plane. The box is
    ``2 * half_height`` tall, so its centre goes at ``trip_height - half_height``: the top
    face lands exactly at the requested height and the lower part sinks into the ground
    (the box is parked there, not simulated, so that is harmless and gives a tall enough
    step for every sampled height).

    The box is yawed so its local x axis points along ``direction``, i.e. its face is
    perpendicular to travel.
    """
    base = _rows_xy(base_xy)
    vel = _rows_xy(velocity_xy)
    n = base.shape[0]
    if vel.shape[0] != n:
        raise ValueError(f"velocity has {vel.shape[0]} rows, base position has {n}")

    heights = np.broadcast_to(np.asarray(trip_height, dtype=np.float64).reshape(-1), (n,))
    if np.any(~np.isfinite(heights)) or np.any(heights < 0.0):
        raise ValueError(f"trip_height must be finite and >= 0, got {heights}")

    speed = np.linalg.norm(vel, axis=1)
    direction = travel_direction(vel, command_xy, min_speed=min_speed)
    distance = travel_distance(
        speed,
        lead_time=lead_time,
        min_distance=min_distance,
        max_distance=max_distance,
    )

    center = np.zeros((n, 3), dtype=np.float64)
    center[:, :2] = base + distance[:, None] * direction
    center[:, 2] = heights - half_height
    yaw = np.arctan2(direction[:, 1], direction[:, 0])
    return TripPlan(direction=direction, distance=distance, center=center, quat=yaw_quaternion_xyzw(yaw))
