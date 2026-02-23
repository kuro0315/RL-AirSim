from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = PROJECT_ROOT / "rl-airsim"
AIRSIM_RPC_PORT = 41451
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PPO training or a baseline-inspired agent on AirSim Drone Racing Lab."
    )

    parser.add_argument(
        "--agent-mode",
        type=str,
        choices=["ppo", "baseline"],
        default="ppo",
        help="Execution mode: PPO training or baseline-inspired spline agent.",
    )

    parser.add_argument("--drone-name", type=str, default="drone_1")
    parser.add_argument("--level-name", type=str, default="Soccer_Field_Easy")
    parser.add_argument("--race-tier", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--episode-max-steps", type=int, default=1000)
    parser.add_argument("--time-step-sec", type=float, default=0.10)
    parser.add_argument("--gate-pass-threshold", type=float, default=1.8)
    parser.add_argument("--max-vel-xy", type=float, default=8.0)
    parser.add_argument("--max-vel-z", type=float, default=4.0)
    parser.add_argument("--max-yaw-rate-deg", type=float, default=90.0)
    parser.add_argument("--reward-progress-scale", type=float, default=1.0)
    parser.add_argument("--reward-time-penalty", type=float, default=-0.01)
    parser.add_argument("--reward-gate-pass", type=float, default=25.0)
    parser.add_argument("--reward-collision", type=float, default=-120.0)
    parser.add_argument("--reward-success", type=float, default=200.0)
    parser.add_argument(
        "--disable-level-reload",
        action="store_true",
        help="If set, skip simLoadLevel on reset after first initialization.",
    )

    parser.add_argument(
        "--rule-controller",
        type=str,
        default="gate_navigator",
        help="Rule-based controller name or import path.",
    )
    parser.add_argument(
        "--disable-rule-controller",
        action="store_true",
        help="Disable rule-based assist entirely.",
    )
    parser.add_argument(
        "--rule-mix",
        type=float,
        default=0.15,
        help="Blend ratio of rule-based action [0, 1].",
    )
    parser.add_argument("--rule-velocity-gain", type=float, default=0.45)
    parser.add_argument("--rule-z-gain", type=float, default=0.8)
    parser.add_argument("--rule-yaw-gain", type=float, default=2.0)
    parser.add_argument("--rule-min-forward-speed", type=float, default=1.0)

    parser.add_argument("--framework", type=str, choices=["torch", "tf2"], default="torch")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--stop-reward", type=float, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--entropy-coeff", type=float, default=0.0)
    parser.add_argument("--train-batch-size", type=int, default=2048)
    parser.add_argument("--num-env-runners", type=int, default=0)
    parser.add_argument("--num-gpus", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument(
        "--baseline-vel-max",
        type=float,
        default=None,
        help="Override baseline moveOnSpline maximum velocity.",
    )
    parser.add_argument(
        "--baseline-acc-max",
        type=float,
        default=None,
        help="Override baseline moveOnSpline maximum acceleration.",
    )

    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument(
        "--sim-launch-mode",
        type=str,
        choices=["gui", "nodisplay"],
        default="gui",
        help=(
            "Simulator launch preset. "
            "'nodisplay' applies ViewMode=NoDisplay and launches with -NoVSync -BENCHMARK."
        ),
    )
    parser.add_argument(
        "--auto-launch-sim",
        action="store_true",
        help="Auto-launch simulator when RPC server is not reachable.",
    )
    return parser.parse_args()


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

    client = _connect_baseline_client(
        airsim_module=airsim,
        sim_launch_mode=args.sim_launch_mode,
    )

    drone_name = _resolve_drone_name(client, args.drone_name)

    client.simLoadLevel(args.level_name)
    client.confirmConnection()
    time.sleep(2.0)

    client.simResetRace()
    effective_tier = _start_race_with_fallback(client, args.race_tier)

    _set_api_control(client, drone_name, True)
    _set_armed(client, drone_name, True)

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
    client.setTrajectoryTrackerGains(gains, vehicle_name=drone_name)
    time.sleep(0.2)

    try:
        client.takeoffAsync(vehicle_name=drone_name).join()
    except TypeError:
        client.takeoffAsync().join()

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

    future = client.moveOnSplineAsync(
        gate_positions,
        vel_max=vel_max,
        acc_max=acc_max,
        add_position_constraint=True,
        add_velocity_constraint=False,
        add_acceleration_constraint=False,
        viz_traj=True,
        viz_traj_color_rgba=[1.0, 0.0, 0.0, 1.0],
        vehicle_name=drone_name,
    )
    future.join()
    print("[baseline] Finished.")

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

    if not _wait_for_airsim_rpc(timeout_sec=timeout_sec, poll_sec=2.0):
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
            print(
                f"[baseline] Configured drone_name '{candidate}' is unavailable; "
                "using default vehicle."
            )
            return ""
        raise


def _start_race_with_fallback(client: Any, requested_tier: int) -> int:
    try:
        client.simStartRace(requested_tier)
        setattr(client, "race_tier", requested_tier)
        return requested_tier
    except Exception as exc:
        text = str(exc)
        missing_competitor = "Vehicle API for 'drone_" in text and "is not available" in text
        if missing_competitor:
            try:
                # Bypass client-side competitor initialization (drone_2) and call server RPC directly.
                client.client.call("simStartRace", requested_tier)
                setattr(client, "race_tier", requested_tier)
                print(
                    f"[baseline] simStartRace(tier={requested_tier}) wrapper failed due to missing "
                    "competitor vehicle; started requested tier via raw RPC."
                )
                return requested_tier
            except Exception as raw_exc:
                if requested_tier != 2:
                    print(
                        f"[baseline] simStartRace(tier={requested_tier}) raw RPC failed; "
                        "falling back to tier=2."
                    )
                    client.simStartRace(2)
                    setattr(client, "race_tier", 2)
                    return 2
                raise raw_exc
        raise


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
    gate_names_sorted_bad = sorted(gate_names_unsorted)
    gate_indices_bad = [int(gate_name.split("_")[0][4:]) for gate_name in gate_names_sorted_bad]
    gate_indices_correct = sorted(
        range(len(gate_indices_bad)), key=lambda idx: gate_indices_bad[idx]
    )
    gate_names_sorted = [gate_names_sorted_bad[idx] for idx in gate_indices_correct]

    positions: list[Any] = []
    for gate_name in gate_names_sorted:
        pose = client.simGetObjectPose(gate_name)
        retries = 0
        while (
            math.isnan(pose.position.x_val)
            or math.isnan(pose.position.y_val)
            or math.isnan(pose.position.z_val)
        ) and retries < 10:
            retries += 1
            pose = client.simGetObjectPose(gate_name)
        if (
            math.isnan(pose.position.x_val)
            or math.isnan(pose.position.y_val)
            or math.isnan(pose.position.z_val)
        ):
            raise RuntimeError(
                f"Gate pose for {gate_name} is invalid after {retries} retries."
            )
        positions.append(pose.position)
    return positions


def _baseline_motion_limits(
    level_name: str, vel_override: float | None, acc_override: float | None
) -> tuple[float, float]:
    if vel_override is not None and acc_override is not None:
        return float(vel_override), float(acc_override)

    defaults = {
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
    vel_default, acc_default = defaults.get(level_name, (20.0, 10.0))
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
