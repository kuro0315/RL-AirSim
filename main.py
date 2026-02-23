from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = PROJECT_ROOT / "rl-airsim"
AIRSIM_RPC_PORT = 41451
LEVEL_LOAD_SLEEP_SEC = 2.0
RPC_WAIT_POLL_SEC = 2.0
MAX_GATE_POSE_TRIALS = 10
BASELINE_VIZ_TRAJ_COLOR_RGBA = [1.0, 0.0, 0.0, 1.0]
BASELINE_TRAJECTORY_GAINS = {
    "kp_cross_track": 5.0,
    "kd_cross_track": 0.0,
    "kp_vel_cross_track": 3.0,
    "kd_vel_cross_track": 0.0,
    "kp_along_track": 0.4,
    "kd_along_track": 0.0,
    "kp_vel_along_track": 0.04,
    "kd_vel_along_track": 0.0,
    "kp_z_track": 2.0,
    "kd_z_track": 0.0,
    "kp_vel_z": 0.4,
    "kd_vel_z": 0.0,
    "kp_yaw": 3.0,
    "kd_yaw": 0.1,
}
BASELINE_LEVEL_MOTION_LIMITS = {
    "Soccer_Field_Easy": (30.0, 15.0),
    "Soccer_Field_Medium": (30.0, 15.0),
    "ZhangJiaJie_Medium": (30.0, 15.0),
    "Qualifier_Tier_1": (30.0, 15.0),
    "Qualifier_Tier_2": (30.0, 15.0),
    "Qualifier_Tier_3": (30.0, 15.0),
    "Final_Tier_1": (30.0, 15.0),
    "Final_Tier_2": (30.0, 15.0),
    "Final_Tier_3": (30.0, 15.0),
    "Building99_Hard": (4.0, 1.0),
}
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


ALLOWED_AGENT_MODES = {"ppo", "baseline"}
ALLOWED_SIM_LAUNCH_MODES = {"gui", "nodisplay"}
ALLOWED_FRAMEWORKS = {"torch", "tf2"}
ALLOWED_IMG_BENCHMARK_TYPES = {"simGetImage", "simGetImages"}

DEFAULT_MAIN_CONFIG: dict[str, Any] = {
    "agent_mode": "ppo",
    "runtime": {
        "sim_launch_mode": "gui",
        "auto_launch_sim": False,
    },
    "env": {
        "drone_name": "drone_1",
        "level_name": "Soccer_Field_Easy",
        "race_tier": 1,
        "episode_max_steps": 1000,
        "time_step_sec": 0.10,
        "gate_pass_threshold": 1.8,
        "max_vel_xy": 8.0,
        "max_vel_z": 4.0,
        "max_yaw_rate_deg": 90.0,
        "reward_progress_scale": 1.0,
        "reward_time_penalty": -0.01,
        "reward_gate_pass": 25.0,
        "reward_collision": -120.0,
        "reward_success": 200.0,
        "disable_level_reload": False,
    },
    "rule_assist": {
        "rule_controller": "gate_navigator",
        "disable_rule_controller": False,
        "rule_mix": 0.15,
        "rule_velocity_gain": 0.45,
        "rule_z_gain": 0.8,
        "rule_yaw_gain": 2.0,
        "rule_min_forward_speed": 1.0,
    },
    "image_benchmark": {
        "enable_image_benchmark": False,
        "img_benchmark_type": "simGetImages",
        "img_benchmark_period_sec": 0.05,
        "img_benchmark_camera_name": "fpv_cam",
    },
    "training": {
        "framework": "torch",
        "iterations": 50,
        "stop_reward": None,
        "lr": 3e-4,
        "gamma": 0.99,
        "lambda_": 0.95,
        "clip_param": 0.2,
        "entropy_coeff": 0.0,
        "train_batch_size": 2048,
        "num_env_runners": 0,
        "num_gpus": 0.0,
        "seed": None,
        "checkpoint_dir": None,
    },
    "baseline": {
        "baseline_vel_max": None,
        "baseline_acc_max": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PPO training or baseline via YAML configuration."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML config file (example: configs/main.sample.yaml).",
    )
    cli_args = parser.parse_args()
    return _load_args_from_yaml(cli_args.config)


def _load_args_from_yaml(config_path: Path) -> argparse.Namespace:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(
            "PyYAML is required. Install it with `pip install pyyaml`."
        ) from exc

    resolved_path = config_path.expanduser()
    if not resolved_path.is_absolute():
        resolved_path = (Path.cwd() / resolved_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Config root must be a mapping/object: {resolved_path}"
        )

    merged = _deep_merge_config(DEFAULT_MAIN_CONFIG, loaded)
    _validate_main_config(merged)
    return _config_dict_to_namespace(merged)


def _deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key not in merged:
            raise KeyError(f"Unknown config key: {key}")
        base_value = merged[key]
        if isinstance(base_value, dict):
            if not isinstance(value, dict):
                raise TypeError(
                    f"Config key '{key}' must be an object/mapping."
                )
            merged[key] = _deep_merge_config(base_value, value)
        else:
            merged[key] = value
    return merged


def _validate_main_config(config: dict[str, Any]) -> None:
    if config["agent_mode"] not in ALLOWED_AGENT_MODES:
        raise ValueError(
            f"agent_mode must be one of {sorted(ALLOWED_AGENT_MODES)}"
        )
    if config["runtime"]["sim_launch_mode"] not in ALLOWED_SIM_LAUNCH_MODES:
        raise ValueError(
            f"runtime.sim_launch_mode must be one of {sorted(ALLOWED_SIM_LAUNCH_MODES)}"
        )
    if int(config["env"]["race_tier"]) not in {1, 2, 3}:
        raise ValueError("env.race_tier must be one of [1, 2, 3]")
    if config["training"]["framework"] not in ALLOWED_FRAMEWORKS:
        raise ValueError(
            f"training.framework must be one of {sorted(ALLOWED_FRAMEWORKS)}"
        )
    if config["image_benchmark"]["img_benchmark_type"] not in ALLOWED_IMG_BENCHMARK_TYPES:
        raise ValueError(
            "image_benchmark.img_benchmark_type must be one of "
            f"{sorted(ALLOWED_IMG_BENCHMARK_TYPES)}"
        )


def _config_dict_to_namespace(config: dict[str, Any]) -> argparse.Namespace:
    runtime = config["runtime"]
    env = config["env"]
    rule = config["rule_assist"]
    image = config["image_benchmark"]
    training = config["training"]
    baseline = config["baseline"]
    return argparse.Namespace(
        agent_mode=str(config["agent_mode"]),
        sim_launch_mode=str(runtime["sim_launch_mode"]),
        auto_launch_sim=bool(runtime["auto_launch_sim"]),
        drone_name=str(env["drone_name"]),
        level_name=str(env["level_name"]),
        race_tier=int(env["race_tier"]),
        episode_max_steps=int(env["episode_max_steps"]),
        time_step_sec=float(env["time_step_sec"]),
        gate_pass_threshold=float(env["gate_pass_threshold"]),
        max_vel_xy=float(env["max_vel_xy"]),
        max_vel_z=float(env["max_vel_z"]),
        max_yaw_rate_deg=float(env["max_yaw_rate_deg"]),
        reward_progress_scale=float(env["reward_progress_scale"]),
        reward_time_penalty=float(env["reward_time_penalty"]),
        reward_gate_pass=float(env["reward_gate_pass"]),
        reward_collision=float(env["reward_collision"]),
        reward_success=float(env["reward_success"]),
        disable_level_reload=bool(env["disable_level_reload"]),
        rule_controller=(
            None
            if rule["rule_controller"] is None
            else str(rule["rule_controller"])
        ),
        disable_rule_controller=bool(rule["disable_rule_controller"]),
        rule_mix=float(rule["rule_mix"]),
        rule_velocity_gain=float(rule["rule_velocity_gain"]),
        rule_z_gain=float(rule["rule_z_gain"]),
        rule_yaw_gain=float(rule["rule_yaw_gain"]),
        rule_min_forward_speed=float(rule["rule_min_forward_speed"]),
        enable_image_benchmark=bool(image["enable_image_benchmark"]),
        img_benchmark_type=str(image["img_benchmark_type"]),
        img_benchmark_period_sec=float(image["img_benchmark_period_sec"]),
        img_benchmark_camera_name=str(image["img_benchmark_camera_name"]),
        framework=str(training["framework"]),
        iterations=int(training["iterations"]),
        stop_reward=(
            None
            if training["stop_reward"] is None
            else float(training["stop_reward"])
        ),
        lr=float(training["lr"]),
        gamma=float(training["gamma"]),
        lambda_=float(training["lambda_"]),
        clip_param=float(training["clip_param"]),
        entropy_coeff=float(training["entropy_coeff"]),
        train_batch_size=int(training["train_batch_size"]),
        num_env_runners=int(training["num_env_runners"]),
        num_gpus=float(training["num_gpus"]),
        seed=(None if training["seed"] is None else int(training["seed"])),
        checkpoint_dir=(
            None
            if training["checkpoint_dir"] in (None, "")
            else str(training["checkpoint_dir"])
        ),
        baseline_vel_max=(
            None
            if baseline["baseline_vel_max"] is None
            else float(baseline["baseline_vel_max"])
        ),
        baseline_acc_max=(
            None
            if baseline["baseline_acc_max"] is None
            else float(baseline["baseline_acc_max"])
        ),
    )


def build_env_config(args: argparse.Namespace) -> dict[str, Any]:
    rule_controller = None if args.disable_rule_controller else args.rule_controller
    rule_mixing_ratio = 0.0 if args.disable_rule_controller else float(args.rule_mix)

    return {
        "drone_name": args.drone_name,
        "level_name": args.level_name,
        "race_tier": args.race_tier,
        "load_level_on_reset": not args.disable_level_reload,
        "time_step_sec": args.time_step_sec,
        "episode_max_steps": args.episode_max_steps,
        "gate_pass_threshold": args.gate_pass_threshold,
        "max_vel_xy": args.max_vel_xy,
        "max_vel_z": args.max_vel_z,
        "max_yaw_rate_deg": args.max_yaw_rate_deg,
        "reward_progress_scale": args.reward_progress_scale,
        "reward_time_penalty": args.reward_time_penalty,
        "reward_gate_pass": args.reward_gate_pass,
        "reward_collision": args.reward_collision,
        "reward_success": args.reward_success,
        "rule_based_controller": rule_controller,
        "rule_mixing_ratio": rule_mixing_ratio,
        "rule_controller_config": {
            "max_vel_xy": args.max_vel_xy,
            "max_vel_z": args.max_vel_z,
            "max_yaw_rate_deg": args.max_yaw_rate_deg,
            "velocity_gain": args.rule_velocity_gain,
            "z_gain": args.rule_z_gain,
            "yaw_gain": args.rule_yaw_gain,
            "min_forward_speed": args.rule_min_forward_speed,
        },
        "enable_image_benchmark": bool(args.enable_image_benchmark),
        "image_benchmark_config": {
            "benchmark_type": args.img_benchmark_type,
            "period_sec": args.img_benchmark_period_sec,
            "camera_name": args.img_benchmark_camera_name,
        },
    }


def build_training_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "framework": args.framework,
        "lr": args.lr,
        "gamma": args.gamma,
        "lambda_": args.lambda_,
        "clip_param": args.clip_param,
        "entropy_coeff": args.entropy_coeff,
        "train_batch_size": args.train_batch_size,
        "num_env_runners": args.num_env_runners,
        "num_gpus": args.num_gpus,
    }
    if args.seed is not None:
        config["seed"] = args.seed
    return config


def run_training(args: argparse.Namespace) -> None:
    import ray
    from adrl_agent import build_ppo_algorithm, extract_training_metrics

    if args.auto_launch_sim:
        _ensure_airsim_server(
            sim_launch_mode=args.sim_launch_mode,
            timeout_sec=90.0,
            log_prefix="ppo",
        )

    env_config = build_env_config(args)
    training_config = build_training_config(args)

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    algorithm = build_ppo_algorithm(
        env_config=env_config,
        training_config=training_config,
    )

    try:
        for iteration in range(1, args.iterations + 1):
            result = algorithm.train()
            metrics = extract_training_metrics(result)

            reward_mean = metrics["episode_reward_mean"]
            reward_text = f"{reward_mean:.3f}" if not math.isnan(reward_mean) else "nan"
            print(
                f"[iter {iteration:04d}] "
                f"reward_mean={reward_text} "
                f"episode_len_mean={metrics['episode_len_mean']:.2f} "
                f"sampled_steps={int(metrics['sampled_steps'])}"
            )

            if args.stop_reward is not None and not math.isnan(reward_mean):
                if reward_mean >= args.stop_reward:
                    print(
                        f"Stop condition reached: episode_reward_mean={reward_mean:.3f} "
                        f">= {args.stop_reward:.3f}"
                    )
                    break

        if args.checkpoint_dir:
            checkpoint = algorithm.save(args.checkpoint_dir)
            print(f"Saved checkpoint: {checkpoint}")
    finally:
        algorithm.stop()
        ray.shutdown()


def run_baseline_agent(args: argparse.Namespace) -> None:
    from adrl_env.airsim_env import _patch_msgpackrpc_compatibility

    _patch_msgpackrpc_compatibility()
    import airsimdroneracinglab as airsim

    client = _connect_baseline_client(airsim_module=airsim, sim_launch_mode=args.sim_launch_mode)
    effective_tier = _reset_world_and_start_race(client, args.level_name, args.race_tier)
    drone_name = _resolve_drone_name(client, args.drone_name)
    _initialize_baseline_drone(client, airsim, drone_name)

    gate_positions = _collect_sorted_gate_positions(client)
    if not gate_positions:
        raise RuntimeError("No gates were found in the current level.")

    vel_max, acc_max = _baseline_motion_limits(
        args.level_name,
        args.baseline_vel_max,
        args.baseline_acc_max,
    )
    print(
        f"[baseline] level={args.level_name} tier={effective_tier} "
        f"gates={len(gate_positions)} vel_max={vel_max:.2f} acc_max={acc_max:.2f}"
    )
    print("[baseline] Following all gates with moveOnSplineAsync...")
    _run_baseline_spline(
        client=client,
        drone_name=drone_name,
        gate_positions=gate_positions,
        vel_max=vel_max,
        acc_max=acc_max,
    )
    print("[baseline] Finished.")
    _shutdown_baseline_drone(client, drone_name)


def _reset_world_and_start_race(client: Any, level_name: str, race_tier: int) -> int:
    client.simLoadLevel(level_name)
    client.confirmConnection()
    time.sleep(LEVEL_LOAD_SLEEP_SEC)
    client.simResetRace()
    return _start_race_with_fallback(client, race_tier)


def _initialize_baseline_drone(client: Any, airsim_module: Any, drone_name: str) -> None:
    _set_api_control(client, drone_name, True)
    _set_armed(client, drone_name, True)
    gains = airsim_module.TrajectoryTrackerGains(**BASELINE_TRAJECTORY_GAINS)
    client.setTrajectoryTrackerGains(gains, vehicle_name=drone_name)
    time.sleep(0.2)
    _takeoff(client, drone_name)


def _takeoff(client: Any, drone_name: str) -> None:
    try:
        client.takeoffAsync(vehicle_name=drone_name).join()
    except TypeError:
        client.takeoffAsync().join()


def _run_baseline_spline(
    client: Any,
    drone_name: str,
    gate_positions: list[Any],
    vel_max: float,
    acc_max: float,
) -> None:
    future = client.moveOnSplineAsync(
        gate_positions,
        vel_max=vel_max,
        acc_max=acc_max,
        add_position_constraint=True,
        add_velocity_constraint=False,
        add_acceleration_constraint=False,
        viz_traj=True,
        viz_traj_color_rgba=BASELINE_VIZ_TRAJ_COLOR_RGBA,
        vehicle_name=drone_name,
    )
    future.join()


def _shutdown_baseline_drone(client: Any, drone_name: str) -> None:
    try:
        _set_armed(client, drone_name, False)
        _set_api_control(client, drone_name, False)
    except Exception:
        pass


def _connect_baseline_client(airsim_module: Any, sim_launch_mode: str):
    _ensure_airsim_server(
        sim_launch_mode=sim_launch_mode,
        timeout_sec=90.0,
        log_prefix="baseline",
    )
    client = airsim_module.MultirotorClient()
    client.confirmConnection()
    return client


def _ensure_airsim_server(sim_launch_mode: str, timeout_sec: float, log_prefix: str) -> None:
    if _is_airsim_rpc_open():
        return

    launched_exe = _try_launch_airsim_simulator(sim_launch_mode)
    if launched_exe is None:
        raise RuntimeError(
            "Could not connect to AirSim server on port 41451 and auto-launch failed. "
            "Start ADRL and try again."
        )

    print(
        f"[{log_prefix}] AirSim RPC server was unreachable. "
        f"Launched simulator: {launched_exe} (mode={sim_launch_mode})"
    )
    print(f"[{log_prefix}] Waiting for RPC server (port {AIRSIM_RPC_PORT})...")

    if not _wait_for_airsim_rpc(timeout_sec=timeout_sec, poll_sec=RPC_WAIT_POLL_SEC):
        raise RuntimeError(
            f"AirSim server did not become available within {int(timeout_sec)} seconds "
            "after auto-launch."
        )

    print(f"[{log_prefix}] RPC server is available.")


def _is_airsim_rpc_open(host: str = "127.0.0.1", port: int = AIRSIM_RPC_PORT) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _wait_for_airsim_rpc(timeout_sec: float, poll_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if _is_airsim_rpc_open():
            return True
        time.sleep(poll_sec)
    return False


def _try_launch_airsim_simulator(sim_launch_mode: str) -> Path | None:
    candidates = [
        PROJECT_ROOT / "ADRL" / "ADRL" / "Binaries" / "Win64" / "ADRL.exe",
        PROJECT_ROOT / "ADRL" / "ADRL.exe",
    ]

    launch_args = ["-windowed"]
    if sim_launch_mode == "nodisplay":
        _apply_nodisplay_settings()
        launch_args.extend(["-NoVSync", "-BENCHMARK"])

    for exe_path in candidates:
        if not exe_path.exists():
            continue
        try:
            subprocess.Popen(
                [str(exe_path), *launch_args],
                cwd=str(exe_path.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return exe_path
        except Exception:
            continue
    return None


def _apply_nodisplay_settings() -> Path | None:
    settings_dst = Path.home() / "Documents" / "AirSim" / "settings.json"
    settings_dst.parent.mkdir(parents=True, exist_ok=True)

    settings_src = (
        PROJECT_ROOT
        / "reference"
        / "AirSim-Drone-Racing-Lab"
        / "settings"
        / "settings_no_view.json"
    )

    data: dict[str, Any] = {}
    if settings_dst.exists():
        try:
            with settings_dst.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}

    # Fallback defaults if there is no user settings yet.
    if not data and settings_src.exists():
        try:
            with settings_src.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}
    if not data:
        data = {"SettingsVersion": 1.2, "SimMode": "Multirotor"}

    # Preserve user-defined vehicle/camera settings and only switch viewport behavior.
    data["ViewMode"] = "NoDisplay"
    with settings_dst.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return settings_dst


def _resolve_drone_name(client: Any, configured_name: str) -> str:
    candidate = str(configured_name or "")
    if not candidate:
        return ""
    try:
        client.isApiControlEnabled(vehicle_name=candidate)
        return candidate
    except Exception as exc:
        text = str(exc)
        if "Vehicle API for" in text and "is not available" in text:
            return ""
        raise


def _start_race_with_fallback(client: Any, requested_tier: int) -> int:
    mode, wrapper_error, raw_error = _attempt_start_race(client, requested_tier)
    if mode is not None:
        if mode == "raw":
            print(
                f"[baseline] simStartRace(tier={requested_tier}) wrapper failed; "
                "started requested tier via raw RPC."
            )
        return requested_tier

    if requested_tier != 2:
        print(
            f"[baseline] simStartRace(tier={requested_tier}) failed via wrapper/raw RPC; "
            "trying fallback tier=2."
        )
        fallback_mode, fallback_wrapper_error, fallback_raw_error = _attempt_start_race(client, 2)
        if fallback_mode is not None:
            if fallback_mode == "raw":
                print("[baseline] simStartRace fallback succeeded via raw RPC (tier=2).")
            return 2

        raise RuntimeError(
            "Failed to start race. "
            f"requested_tier={requested_tier}, wrapper_error={wrapper_error}, "
            f"raw_error={raw_error}, tier2_wrapper_error={fallback_wrapper_error}, "
            f"tier2_raw_error={fallback_raw_error}"
        ) from fallback_raw_error

    raise RuntimeError(
        "Failed to start race at tier=2 via both wrapper and raw RPC. "
        f"wrapper_error={wrapper_error}, raw_error={raw_error}"
    ) from raw_error


def _attempt_start_race(client: Any, tier: int) -> tuple[str | None, Exception | None, Exception | None]:
    try:
        client.simStartRace(tier)
        setattr(client, "race_tier", tier)
        return "wrapper", None, None
    except Exception as wrapper_error:
        try:
            # Bypass client-side competitor initialization (drone_2) and call server RPC directly.
            client.client.call("simStartRace", tier)
            setattr(client, "race_tier", tier)
            return "raw", wrapper_error, None
        except Exception as raw_error:
            return None, wrapper_error, raw_error


def _set_api_control(client: Any, vehicle_name: str, enabled: bool) -> None:
    if enabled:
        try:
            client.enableApiControl(vehicle_name=vehicle_name)
        except TypeError:
            client.enableApiControl(True, vehicle_name=vehicle_name)
        return

    disable_fn = getattr(client, "disableApiControl", None)
    if callable(disable_fn):
        try:
            disable_fn(vehicle_name=vehicle_name)
            return
        except TypeError:
            pass

    try:
        client.enableApiControl(False, vehicle_name=vehicle_name)
    except TypeError:
        pass


def _set_armed(client: Any, vehicle_name: str, armed: bool) -> None:
    if armed:
        try:
            client.arm(vehicle_name=vehicle_name)
        except TypeError:
            client.arm(True, vehicle_name=vehicle_name)
        return

    disarm_fn = getattr(client, "disarm", None)
    if callable(disarm_fn):
        try:
            disarm_fn(vehicle_name=vehicle_name)
            return
        except TypeError:
            pass

    try:
        client.arm(False, vehicle_name=vehicle_name)
    except TypeError:
        pass


def _collect_sorted_gate_positions(client: Any) -> list[Any]:
    gate_names_unsorted = client.simListSceneObjects("Gate.*")
    gate_names_sorted = sorted(gate_names_unsorted, key=_gate_sort_key)

    positions: list[Any] = []
    for gate_name in gate_names_sorted:
        pose = client.simGetObjectPose(gate_name)
        retries = 0
        while not _is_valid_pose_position(pose) and retries < MAX_GATE_POSE_TRIALS:
            retries += 1
            pose = client.simGetObjectPose(gate_name)
        if not _is_valid_pose_position(pose):
            raise RuntimeError(
                f"Gate pose for {gate_name} is invalid after {retries} retries."
            )
        positions.append(pose.position)
    return positions


def _gate_sort_key(gate_name: str) -> tuple[int, str]:
    try:
        return int(gate_name.split("_")[0][4:]), gate_name
    except (IndexError, ValueError):
        return 10**9, gate_name


def _is_valid_pose_position(pose: Any) -> bool:
    return not (
        math.isnan(pose.position.x_val)
        or math.isnan(pose.position.y_val)
        or math.isnan(pose.position.z_val)
    )


def _baseline_motion_limits(
    level_name: str, vel_override: float | None, acc_override: float | None
) -> tuple[float, float]:
    if vel_override is not None and acc_override is not None:
        return float(vel_override), float(acc_override)

    vel_default, acc_default = BASELINE_LEVEL_MOTION_LIMITS.get(level_name, (20.0, 10.0))
    return (
        float(vel_override) if vel_override is not None else vel_default,
        float(acc_override) if acc_override is not None else acc_default,
    )


def main() -> None:
    args = parse_args()
    if args.agent_mode == "baseline":
        run_baseline_agent(args)
        return
    run_training(args)


if __name__ == "__main__":
    main()
