from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compose and validate the exact Hydra overrides used for MBPP OPD.")
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--n-gpus", type=int, required=True)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_dir = Path(args.config_dir).expanduser().resolve()
    config_file = config_dir / "ppo_trainer.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(f"Missing verl trainer config: {config_file}")

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name="ppo_trainer", overrides=args.overrides)

    # Resolving catches broken interpolation in addition to unknown/invalid Hydra keys.
    OmegaConf.resolve(config)
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    global_batch = int(config.data.train_batch_size)
    mini_batch = int(actor.ppo_mini_batch_size)
    micro_batch = int(actor.ppo_micro_batch_size_per_gpu)
    rollout_n = int(rollout.n)

    expected_global_batch = micro_batch * args.gradient_accumulation_steps * args.n_gpus
    if global_batch != expected_global_batch:
        raise ValueError(
            f"Batch invariant failed: {global_batch=} != {micro_batch=} * "
            f"gradient_accumulation_steps={args.gradient_accumulation_steps} * n_gpus={args.n_gpus}"
        )
    if mini_batch * rollout_n % (micro_batch * args.n_gpus) != 0:
        raise ValueError("Normalized actor mini-batch is not divisible by the per-device micro-batch")
    effective_grad_acc = mini_batch * rollout_n // (micro_batch * args.n_gpus)
    if effective_grad_acc != args.gradient_accumulation_steps:
        raise ValueError(f"Effective actor gradient accumulation is {effective_grad_acc}, not the requested value")
    if config.algorithm.adv_estimator != "token_reward_direct":
        raise ValueError("MBPP OPD must use token_reward_direct")
    if config.algorithm.use_kl_in_reward or actor.use_kl_loss:
        raise ValueError("Extra KL reward/loss must stay disabled because token_reward_direct supplies OPD supervision")
    if not config.reward_model.enable:
        raise ValueError("The frozen teacher worker must be enabled")
    if int(config.actor_rollout_ref.model.get("lora_rank", 0)) != 0:
        raise ValueError("This entrypoint is intended for full-parameter student training, not LoRA")
    if config.actor_rollout_ref.model.path == config.reward_model.model.path:
        raise ValueError("Student and teacher model paths must be different")
    if not actor.use_dynamic_bsz:
        rollout_legacy_micro = rollout.log_prob_micro_batch_size
        rollout_device_micro = rollout.log_prob_micro_batch_size_per_gpu
        if (rollout_legacy_micro is None) == (rollout_device_micro is None):
            raise ValueError(
                "Fixed actor batching requires exactly one rollout log-prob micro-batch setting, as enforced by verl"
            )
    if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
        reward_legacy_micro = config.reward_model.micro_batch_size
        reward_device_micro = config.reward_model.micro_batch_size_per_gpu
        if (reward_legacy_micro is None) == (reward_device_micro is None):
            raise ValueError("The frozen teacher requires exactly one reward-model micro-batch setting")

    # Run verl's own dataclass and batch validation without creating Ray workers or loading model weights.
    from verl.utils.config import validate_config

    use_reference_policy = bool(config.algorithm.use_kl_in_reward or actor.use_kl_loss)
    use_critic = bool(
        config.critic.enable if config.critic.enable is not None else config.algorithm.adv_estimator == "gae"
    )
    validate_config(config, use_reference_policy=use_reference_policy, use_critic=use_critic)

    print(
        "Hydra config passed: "
        f"steps={config.trainer.total_training_steps}, "
        f"batch={config.data.train_batch_size}, "
        f"micro_batch={config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu}, "
        f"grad_acc={effective_grad_acc}, "
        f"student={config.actor_rollout_ref.model.path}, "
        f"teacher={config.reward_model.model.path}"
    )


if __name__ == "__main__":
    main()
