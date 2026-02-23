from __future__ import annotations

import importlib
import math
import time
from typing import Any, Mapping, Optional

import airsimdroneracinglab as airsim
import gymnasium as gym
import numpy as np
from gymnasium import spaces

ENV_NAME = "AirSimDroneRacingEnv-v0"
MAX_GETOBJECTPOSE_TRIALS = 10

DEFAULT_ENV_CONFIG: dict[str, Any] = {
    "drone_name": "drone_1",
    "level_name": "Soccer_Field_Easy",
    "race_tier": 1,
    "load_level_on_reset": True,
    "level_load_sleep_sec": 2.0,
    "time_step_sec": 0.10,
    "episode_max_steps": 1000,
    "takeoff_height": 1.0,
    "max_vel_xy": 8.0,
    "max_vel_z": 4.0,
    "max_yaw_rate_deg": 90.0,
    "angular_velocity_scale": 3.0,
    "relative_position_scale": 30.0,
    "distance_scale": 50.0,
    "gate_pass_threshold": 1.8,
    "reward_progress_scale": 1.0,
    "reward_time_penalty": -0.01,
    "reward_gate_pass": 25.0,
    "reward_collision": -120.0,
    "reward_success": 200.0,
    "rule_based_controller": "gate_navigator",
    "rule_controller_config": {},
    "rule_mixing_ratio": 0.0,
}


def _patch_msgpackrpc_compatibility() -> None:
    """Patch msgpack-rpc-python for msgpack>=1 where encoding args were removed."""
    try:
        import msgpack
        from msgpackrpc.transport import tcp as msgpackrpc_tcp
    except Exception:
        return

    if getattr(msgpackrpc_tcp, "_adrl_msgpack_patch_applied", False):
        return

    def _default_serializer(value: Any):
        return value.to_msgpack()

    def _base_socket_init(self, stream, encodings):
        self._stream = stream

        try:
            self._packer = msgpack.Packer(
                encoding=encodings[0],
                default=_default_serializer,
            )
        except TypeError:
            self._packer = msgpack.Packer(
                default=_default_serializer,
                use_bin_type=True,
            )

        try:
            self._unpacker = msgpack.Unpacker(encoding=encodings[1])
        except TypeError:
            # msgpack>=1 removed `encoding`; decode text payloads for RPC method names.
            self._unpacker = msgpack.Unpacker(raw=False)

    msgpackrpc_tcp.BaseSocket.__init__ = _base_socket_init
    msgpackrpc_tcp._adrl_msgpack_patch_applied = True


_patch_msgpackrpc_compatibility()


def _vec3_to_np(vec3: Any) -> np.ndarray:
    return np.array([vec3.x_val, vec3.y_val, vec3.z_val], dtype=np.float32)


class AirSimDroneRacingEnv(gym.Env):
    """AirSim Drone Racing Lab environment for RLlib/Gymnasium."""

    metadata = {"render_modes": []}

    def __init__(self, env_config: Optional[Mapping[str, Any]] = None):
        if airsim is None:
            raise ImportError(
                "airsimdroneracinglab is required. Install AirSim Drone Racing Lab Python client first."
            )

        merged_config = dict(DEFAULT_ENV_CONFIG)
        if env_config:
            merged_config.update(dict(env_config))
        self.config = merged_config

        self.drone_name = str(self.config["drone_name"])
        self.level_name = str(self.config["level_name"])
        self.race_tier = int(self.config["race_tier"])
        self.time_step_sec = float(self.config["time_step_sec"])
        self.episode_max_steps = int(self.config["episode_max_steps"])
        self.takeoff_height = float(self.config["takeoff_height"])
        self.max_vel_xy = float(self.config["max_vel_xy"])
        self.max_vel_z = float(self.config["max_vel_z"])
        self.max_yaw_rate_deg = float(self.config["max_yaw_rate_deg"])
        self.angular_velocity_scale = float(self.config["angular_velocity_scale"])
        self.relative_position_scale = float(self.config["relative_position_scale"])
        self.distance_scale = float(self.config["distance_scale"])
        self.gate_pass_threshold = float(self.config["gate_pass_threshold"])
        self.reward_progress_scale = float(self.config["reward_progress_scale"])
        self.reward_time_penalty = float(self.config["reward_time_penalty"])
        self.reward_gate_pass = float(self.config["reward_gate_pass"])
        self.reward_collision = float(self.config["reward_collision"])
        self.reward_success = float(self.config["reward_success"])
        self.rule_mixing_ratio = float(self.config["rule_mixing_ratio"])

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32,
        )

        self.client: Any = None
        self._loaded_level_name: Optional[str] = None

        self.current_gate_index = 0
        self.step_count = 0
        self.gate_positions: list[np.ndarray] = []
        self.previous_distance = 0.0

        self.rule_controller = self._load_rule_controller()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)
        self._connect()
        self._reset_race_world(options or {})

        position, linear_velocity, angular_velocity, orientation = self._read_kinematics()
        self.previous_distance = self._distance_to_current_gate(position)
        self.step_count = 0

        observation = self._build_observation(
            position, linear_velocity, angular_velocity, orientation
        )
        info = self._build_info(False, False, False)
        return observation, info

    def step(self, action: np.ndarray):
        if self.client is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        self.step_count += 1
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        action = np.clip(action, -1.0, 1.0)

        pre_position, pre_linear_velocity, _, pre_orientation = self._read_kinematics()
        prev_distance = self._distance_to_current_gate(pre_position)

        applied_action = self._mix_with_rule_controller(
            action, pre_position, pre_linear_velocity, pre_orientation
        )
        self._execute_action(applied_action)

        position, linear_velocity, angular_velocity, orientation = self._read_kinematics()
        curr_distance = self._distance_to_current_gate(position)

        reward = self.reward_time_penalty + self.reward_progress_scale * (
            prev_distance - curr_distance
        )
        terminated = False
        truncated = False
        gate_passed = False

        if self._is_current_gate_passed(curr_distance):
            gate_passed = True
            reward += self.reward_gate_pass
            self.current_gate_index += 1
            if self.current_gate_index >= len(self.gate_positions):
                terminated = True
                reward += self.reward_success

        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_name)
        if getattr(collision_info, "has_collided", False):
            terminated = True
            reward += self.reward_collision

        if self.step_count >= self.episode_max_steps and not terminated:
            truncated = True

        self.previous_distance = curr_distance
        observation = self._build_observation(
            position, linear_velocity, angular_velocity, orientation
        )
        info = self._build_info(
            gate_passed=gate_passed,
            terminated=terminated,
            truncated=truncated,
        )

        return observation, float(reward), terminated, truncated, info

    def close(self):
        if self.client is None:
            return
        try:
            self._set_armed(False)
        except Exception:
            pass
        try:
            self._set_api_control(False)
        except Exception:
            pass

    def _connect(self):
        if self.client is not None:
            return
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self._resolve_drone_name()

    def _resolve_drone_name(self) -> None:
        if not self.drone_name:
            return
        try:
            self.client.isApiControlEnabled(vehicle_name=self.drone_name)
        except Exception as exc:
            error_text = str(exc)
            if "Vehicle API for" in error_text and "is not available" in error_text:
                print(
                    f"Configured drone_name '{self.drone_name}' is unavailable. "
                    "Falling back to default vehicle."
                )
                self.drone_name = ""
                return
            raise

    def _reset_race_world(self, options: Mapping[str, Any]):
        if options.get("level_name"):
            self.level_name = str(options["level_name"])
        if options.get("race_tier"):
            self.race_tier = int(options["race_tier"])

        should_reload = bool(self.config["load_level_on_reset"])
        if options.get("force_reload_level"):
            should_reload = True

        if should_reload and self.level_name != self._loaded_level_name:
            self.client.simLoadLevel(self.level_name)
            self.client.confirmConnection()
            time.sleep(float(self.config["level_load_sleep_sec"]))
            self._loaded_level_name = self.level_name

        self.client.simResetRace()
        self._start_race_with_fallback(self.race_tier)
        self._initialize_drone()
        self._takeoff()
        self.gate_positions = self._get_ground_truth_gate_positions()
        self.current_gate_index = 0

    def _start_race_with_fallback(self, requested_tier: int) -> None:
        try:
            self.client.simStartRace(requested_tier)
            setattr(self.client, "race_tier", requested_tier)
            return
        except Exception as exc:
            error_text = str(exc)
            missing_competitor = (
                "Vehicle API for 'drone_" in error_text and "is not available" in error_text
            )
            if missing_competitor:
                try:
                    # Bypass client-side competitor initialization (drone_2) and call server RPC directly.
                    self.client.client.call("simStartRace", requested_tier)
                    setattr(self.client, "race_tier", requested_tier)
                    print(
                        f"simStartRace(tier={requested_tier}) wrapper failed due to missing "
                        "competitor vehicle. Started requested tier via raw RPC."
                    )
                    return
                except Exception as raw_exc:
                    if requested_tier != 2:
                        print(
                            f"simStartRace(tier={requested_tier}) raw RPC failed. "
                            "Falling back to tier=2."
                        )
                        self.client.simStartRace(2)
                        setattr(self.client, "race_tier", 2)
                        return
                    raise raw_exc
            raise

    def _initialize_drone(self):
        self._set_api_control(True)
        self._set_armed(True)

        gains = airsim.TrajectoryTrackerGains(
            kp_cross_track=5.0,
            kd_cross_track=0.0,
            kp_vel_cross_track=3.0,
            kd_vel_cross_track=0.0,
            kp_along_track=0.4,
            kd_along_track=0.0,
            kp_vel_along_track=0.04,
            kd_vel_along_track=0.0,
            kp_z_track=2.0,
            kd_z_track=0.0,
            kp_vel_z=0.4,
            kd_vel_z=0.0,
            kp_yaw=3.0,
            kd_yaw=0.1,
        )
        self.client.setTrajectoryTrackerGains(gains, vehicle_name=self.drone_name)
        time.sleep(0.2)

    def _set_api_control(self, enabled: bool) -> None:
        if enabled:
            try:
                self.client.enableApiControl(vehicle_name=self.drone_name)
            except TypeError:
                self.client.enableApiControl(True, vehicle_name=self.drone_name)
            return

        disable_fn = getattr(self.client, "disableApiControl", None)
        if callable(disable_fn):
            try:
                disable_fn(vehicle_name=self.drone_name)
                return
            except TypeError:
                pass

        try:
            self.client.enableApiControl(False, vehicle_name=self.drone_name)
        except TypeError:
            # Some client builds only expose enableApiControl(vehicle_name).
            pass

    def _set_armed(self, armed: bool) -> None:
        if armed:
            try:
                self.client.arm(vehicle_name=self.drone_name)
            except TypeError:
                self.client.arm(True, vehicle_name=self.drone_name)
            return

        disarm_fn = getattr(self.client, "disarm", None)
        if callable(disarm_fn):
            try:
                disarm_fn(vehicle_name=self.drone_name)
                return
            except TypeError:
                pass

        try:
            self.client.arm(False, vehicle_name=self.drone_name)
        except TypeError:
            # Some client builds only expose arm(vehicle_name).
            pass

    def _takeoff(self):
        try:
            self.client.takeoffAsync(vehicle_name=self.drone_name).join()
        except TypeError:
            self.client.takeoffAsync().join()

        if self.takeoff_height <= 0.0:
            return

        pose = self.client.simGetVehiclePose(vehicle_name=self.drone_name)
        target_z = pose.position.z_val - self.takeoff_height
        self.client.moveToZAsync(target_z, velocity=2.0, vehicle_name=self.drone_name).join()

    def _get_ground_truth_gate_positions(self) -> list[np.ndarray]:
        gate_names_unsorted = self.client.simListSceneObjects("Gate.*")
        gate_names_sorted_bad = sorted(gate_names_unsorted)
        gate_indices_bad = [
            int(gate_name.split("_")[0][4:]) for gate_name in gate_names_sorted_bad
        ]
        gate_indices_correct = sorted(
            range(len(gate_indices_bad)), key=lambda idx: gate_indices_bad[idx]
        )
        gate_names_sorted = [
            gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct
        ]

        gate_positions: list[np.ndarray] = []
        for gate_name in gate_names_sorted:
            pose = self.client.simGetObjectPose(gate_name)
            retries = 0
            while (
                math.isnan(pose.position.x_val)
                or math.isnan(pose.position.y_val)
                or math.isnan(pose.position.z_val)
            ) and retries < MAX_GETOBJECTPOSE_TRIALS:
                retries += 1
                pose = self.client.simGetObjectPose(gate_name)
            if (
                math.isnan(pose.position.x_val)
                or math.isnan(pose.position.y_val)
                or math.isnan(pose.position.z_val)
            ):
                raise RuntimeError(
                    f"Gate pose for {gate_name} is invalid after {retries} retries."
                )
            gate_positions.append(_vec3_to_np(pose.position))
        return gate_positions

    def _read_kinematics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state = self.client.getMultirotorState(vehicle_name=self.drone_name)
        kine = state.kinematics_estimated
        position = _vec3_to_np(kine.position)
        linear_velocity = _vec3_to_np(kine.linear_velocity)
        angular_velocity = _vec3_to_np(kine.angular_velocity)
        orientation = np.array(
            [
                kine.orientation.w_val,
                kine.orientation.x_val,
                kine.orientation.y_val,
                kine.orientation.z_val,
            ],
            dtype=np.float32,
        )
        return position, linear_velocity, angular_velocity, orientation

    def _build_observation(
        self,
        position: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
        orientation: np.ndarray,
    ) -> np.ndarray:
        gate_position = self._current_gate_position()
        if gate_position is None:
            relative_position = np.zeros(3, dtype=np.float32)
            distance = 0.0
        else:
            relative_position = gate_position - position
            distance = float(np.linalg.norm(relative_position))

        relative_position = np.clip(
            relative_position / max(self.relative_position_scale, 1e-6), -1.0, 1.0
        )

        linear_velocity_norm = np.array(
            [
                linear_velocity[0] / max(self.max_vel_xy, 1e-6),
                linear_velocity[1] / max(self.max_vel_xy, 1e-6),
                linear_velocity[2] / max(self.max_vel_z, 1e-6),
            ],
            dtype=np.float32,
        )
        linear_velocity_norm = np.clip(linear_velocity_norm, -1.0, 1.0)

        angular_velocity_norm = np.clip(
            angular_velocity / max(self.angular_velocity_scale, 1e-6), -1.0, 1.0
        )

        progress = np.array(
            [self.current_gate_index / max(len(self.gate_positions), 1)], dtype=np.float32
        )
        distance_norm = np.array(
            [min(1.0, distance / max(self.distance_scale, 1e-6))], dtype=np.float32
        )

        observation = np.concatenate(
            [
                relative_position.astype(np.float32),
                linear_velocity_norm.astype(np.float32),
                angular_velocity_norm.astype(np.float32),
                np.clip(orientation, -1.0, 1.0).astype(np.float32),
                progress,
                distance_norm,
            ]
        )
        return observation.astype(np.float32)

    def _execute_action(self, action: np.ndarray):
        vx = float(action[0] * self.max_vel_xy)
        vy = float(action[1] * self.max_vel_xy)
        vz = float(action[2] * self.max_vel_z)
        yaw_rate_deg = float(action[3] * self.max_yaw_rate_deg)
        yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate_deg)

        try:
            future = self.client.moveByVelocityAsync(
                vx,
                vy,
                vz,
                duration=self.time_step_sec,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=yaw_mode,
                vehicle_name=self.drone_name,
            )
        except TypeError:
            future = self.client.moveByVelocityAsync(
                vx,
                vy,
                vz,
                self.time_step_sec,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=yaw_mode,
            )
        future.join()

    def _current_gate_position(self) -> Optional[np.ndarray]:
        if self.current_gate_index >= len(self.gate_positions):
            return None
        return self.gate_positions[self.current_gate_index]

    def _distance_to_current_gate(self, position: np.ndarray) -> float:
        gate_position = self._current_gate_position()
        if gate_position is None:
            return 0.0
        return float(np.linalg.norm(gate_position - position))

    def _is_current_gate_passed(self, distance_to_current_gate: float) -> bool:
        if self.current_gate_index >= len(self.gate_positions):
            return False
        return distance_to_current_gate <= self.gate_pass_threshold

    def _build_info(self, gate_passed: bool, terminated: bool, truncated: bool) -> dict[str, Any]:
        return {
            "gate_index": self.current_gate_index,
            "num_gates": len(self.gate_positions),
            "gate_passed": gate_passed,
            "distance_to_gate": self.previous_distance,
            "terminated": terminated,
            "truncated": truncated,
            "rule_mixing_ratio": self.rule_mixing_ratio,
        }

    def _load_rule_controller(self):
        controller_name = self.config.get("rule_based_controller")
        if controller_name in (None, "", "none"):
            return None

        module = importlib.import_module("adrl_agent.rule_based_controller")
        build_fn = getattr(module, "build_rule_based_controller")
        return build_fn(controller_name, self.config.get("rule_controller_config", {}))

    def _mix_with_rule_controller(
        self,
        action: np.ndarray,
        position: np.ndarray,
        linear_velocity: np.ndarray,
        orientation: np.ndarray,
    ) -> np.ndarray:
        if self.rule_controller is None:
            return action

        mix = float(np.clip(self.rule_mixing_ratio, 0.0, 1.0))
        if mix <= 0.0:
            return action

        gate_position = self._current_gate_position()
        if gate_position is None:
            return action

        context = {
            "position": position,
            "linear_velocity": linear_velocity,
            "gate_position": gate_position,
            "yaw": self._yaw_from_quaternion(orientation),
            "gate_index": self.current_gate_index,
            "num_gates": len(self.gate_positions),
        }
        rule_action = np.asarray(
            self.rule_controller.compute_action(context), dtype=np.float32
        ).reshape(action.shape)
        rule_action = np.clip(rule_action, -1.0, 1.0)
        blended = (1.0 - mix) * action + mix * rule_action
        return np.clip(blended, -1.0, 1.0)

    @staticmethod
    def _yaw_from_quaternion(orientation: np.ndarray) -> float:
        w, x, y, z = [float(v) for v in orientation]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def env_creator(env_config: Optional[dict[str, Any]] = None) -> AirSimDroneRacingEnv:
    return AirSimDroneRacingEnv(env_config or {})
