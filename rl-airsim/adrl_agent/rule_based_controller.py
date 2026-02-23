from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


@dataclass
class GateNavigatorConfig:
    max_vel_xy: float = 8.0
    max_vel_z: float = 4.0
    max_yaw_rate_deg: float = 90.0
    velocity_gain: float = 0.45
    z_gain: float = 0.8
    yaw_gain: float = 2.0
    min_forward_speed: float = 1.0


class GateNavigatorRuleController:
    """Simple rule-based controller that points velocity toward current gate."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        cfg = dict(config or {})
        self.config = GateNavigatorConfig(
            max_vel_xy=float(cfg.get("max_vel_xy", GateNavigatorConfig.max_vel_xy)),
            max_vel_z=float(cfg.get("max_vel_z", GateNavigatorConfig.max_vel_z)),
            max_yaw_rate_deg=float(
                cfg.get("max_yaw_rate_deg", GateNavigatorConfig.max_yaw_rate_deg)
            ),
            velocity_gain=float(cfg.get("velocity_gain", GateNavigatorConfig.velocity_gain)),
            z_gain=float(cfg.get("z_gain", GateNavigatorConfig.z_gain)),
            yaw_gain=float(cfg.get("yaw_gain", GateNavigatorConfig.yaw_gain)),
            min_forward_speed=float(
                cfg.get("min_forward_speed", GateNavigatorConfig.min_forward_speed)
            ),
        )

    def compute_action(self, context: Mapping[str, Any]) -> np.ndarray:
        position = np.asarray(context["position"], dtype=np.float32)
        linear_velocity = np.asarray(context["linear_velocity"], dtype=np.float32)
        gate_position = np.asarray(context["gate_position"], dtype=np.float32)
        yaw = float(context.get("yaw", 0.0))

        to_gate = gate_position - position
        distance = float(np.linalg.norm(to_gate))
        if distance < 1e-5:
            return np.zeros(4, dtype=np.float32)

        direction = to_gate / distance
        speed_xy = np.clip(
            self.config.min_forward_speed + self.config.velocity_gain * distance,
            self.config.min_forward_speed,
            self.config.max_vel_xy,
        )

        desired_velocity = np.zeros(3, dtype=np.float32)
        desired_velocity[0] = direction[0] * speed_xy
        desired_velocity[1] = direction[1] * speed_xy
        desired_velocity[2] = np.clip(
            to_gate[2] * self.config.z_gain,
            -self.config.max_vel_z,
            self.config.max_vel_z,
        )

        # Use one-step PD-like correction so the heuristic can damp oscillations.
        command_velocity = desired_velocity - 0.3 * linear_velocity
        command_velocity[0] = np.clip(
            command_velocity[0], -self.config.max_vel_xy, self.config.max_vel_xy
        )
        command_velocity[1] = np.clip(
            command_velocity[1], -self.config.max_vel_xy, self.config.max_vel_xy
        )
        command_velocity[2] = np.clip(
            command_velocity[2], -self.config.max_vel_z, self.config.max_vel_z
        )

        action = np.zeros(4, dtype=np.float32)
        action[0] = command_velocity[0] / max(self.config.max_vel_xy, 1e-6)
        action[1] = command_velocity[1] / max(self.config.max_vel_xy, 1e-6)
        action[2] = command_velocity[2] / max(self.config.max_vel_z, 1e-6)

        desired_yaw = math.atan2(float(direction[1]), float(direction[0]))
        yaw_error = _wrap_pi(desired_yaw - yaw)
        yaw_rate = np.clip(
            self.config.yaw_gain * yaw_error,
            -math.radians(self.config.max_yaw_rate_deg),
            math.radians(self.config.max_yaw_rate_deg),
        )
        action[3] = yaw_rate / max(math.radians(self.config.max_yaw_rate_deg), 1e-6)

        return np.clip(action, -1.0, 1.0).astype(np.float32)


def build_rule_based_controller(
    controller_name: Optional[str], config: Optional[Mapping[str, Any]] = None
):
    if controller_name in (None, "", "none", "off"):
        return None

    normalized = str(controller_name).strip().lower()
    if normalized in {"gate", "gate_navigator", "heuristic"}:
        return GateNavigatorRuleController(config)

    if "." in normalized:
        return _build_controller_from_import_path(str(controller_name), config)

    raise ValueError(f"Unknown rule-based controller: {controller_name}")


def _build_controller_from_import_path(
    import_path: str, config: Optional[Mapping[str, Any]]
):
    module_name, _, attr_name = import_path.rpartition(".")
    if not module_name:
        raise ValueError(
            "Custom controller must be provided as '<module>.<class_or_factory>'."
        )

    module = importlib.import_module(module_name)
    builder = getattr(module, attr_name)

    if isinstance(builder, type):
        return builder(config or {})
    return builder(config or {})


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
