from __future__ import annotations

from typing import Any, Mapping, Optional

from adrl_env.airsim_env import ENV_NAME, env_creator


DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "fcnet_hiddens": [256, 256],
    "fcnet_activation": "relu",
}


def build_ppo_algorithm(
    env_config: Optional[Mapping[str, Any]] = None,
    training_config: Optional[Mapping[str, Any]] = None,
):
    """Build a PPO algorithm instance configured for AirSim racing."""

    try:
        from ray.rllib.algorithms.ppo import PPOConfig
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("ray[rllib] is required to train with RLlib.") from exc

    _register_env()

    env_conf = dict(env_config or {})
    train_conf = dict(training_config or {})

    config = PPOConfig()
    config = config.environment(env=ENV_NAME, env_config=env_conf)
    config = config.framework(str(train_conf.get("framework", "torch")))
    config = _apply_runner_config(config, train_conf)
    config = config.resources(num_gpus=float(train_conf.get("num_gpus", 0.0)))
    config = config.training(
        lr=float(train_conf.get("lr", 3e-4)),
        gamma=float(train_conf.get("gamma", 0.99)),
        lambda_=float(train_conf.get("lambda_", 0.95)),
        clip_param=float(train_conf.get("clip_param", 0.2)),
        entropy_coeff=float(train_conf.get("entropy_coeff", 0.0)),
        train_batch_size=int(train_conf.get("train_batch_size", 2048)),
        model=dict(train_conf.get("model", DEFAULT_MODEL_CONFIG)),
    )

    if "seed" in train_conf:
        config = config.debugging(seed=int(train_conf["seed"]))

    return config.build()


def extract_training_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    episode_reward_mean = float(result.get("episode_reward_mean", float("nan")))
    episode_len_mean = float(result.get("episode_len_mean", float("nan")))
    sampled_steps = float(
        result.get(
            "num_env_steps_sampled_lifetime",
            result.get("num_env_steps_sampled", result.get("timesteps_total", 0.0)),
        )
    )
    return {
        "episode_reward_mean": episode_reward_mean,
        "episode_len_mean": episode_len_mean,
        "sampled_steps": sampled_steps,
    }


def _register_env():
    from ray.tune.registry import register_env

    register_env(ENV_NAME, lambda cfg: env_creator(cfg))


def _apply_runner_config(config, train_conf: Mapping[str, Any]):
    num_env_runners = int(train_conf.get("num_env_runners", 0))
    if hasattr(config, "env_runners"):
        return config.env_runners(num_env_runners=num_env_runners)
    return config.rollouts(num_rollout_workers=num_env_runners)
