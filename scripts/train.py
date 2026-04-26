#!/usr/bin/env python3
"""
train.py — Entry point for MARL strategy game training.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/4agent_team.yaml
    python scripts/train.py --config configs/default.yaml --total_timesteps 1000000
    python scripts/train.py --resume checkpoints/agent_0_ep100.pt --tag 100
"""

import argparse
import logging
import sys
import os
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo_agent import PPOConfig
from training.trainer import MARLTrainer
from training.self_play import SelfPlayManager


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log"),
        ],
    )


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_ppo_config(cfg: dict, device_override: str = None) -> PPOConfig:
    ppo = cfg.get("ppo", {})
    device = device_override or ppo.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logging.getLogger(__name__).warning("CUDA not available, falling back to CPU.")
        device = "cpu"
    return PPOConfig(
        lr              = ppo.get("lr",               3e-4),
        gamma           = ppo.get("gamma",            0.99),
        gae_lambda      = ppo.get("gae_lambda",       0.95),
        clip_eps        = ppo.get("clip_eps",         0.2),
        value_clip_eps  = ppo.get("value_clip_eps",   0.2),
        entropy_coef    = ppo.get("entropy_coef",     0.01),
        value_loss_coef = ppo.get("value_loss_coef",  0.5),
        max_grad_norm   = ppo.get("max_grad_norm",    0.5),
        n_epochs        = ppo.get("n_epochs",         8),
        batch_size      = ppo.get("batch_size",       256),
        rollout_steps   = ppo.get("rollout_steps",    2048),
        hidden_dim      = ppo.get("hidden_dim",       256),
        centralized     = ppo.get("centralized",      True),
        normalize_adv   = ppo.get("normalize_adv",   True),
        device          = device,
    )


def main():
    parser = argparse.ArgumentParser(description="Train MARL Strategy Game Agents")
    parser.add_argument("--config",           default="configs/default.yaml")
    parser.add_argument("--total_timesteps",  type=int,   default=None)
    parser.add_argument("--seed",             type=int,   default=None)
    parser.add_argument("--device",           default=None, help="cpu or cuda")
    parser.add_argument("--resume",           default=None, help="Checkpoint path prefix")
    parser.add_argument("--tag",              default=None, help="Checkpoint tag to resume")
    parser.add_argument("--log_level",        default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    log = logging.getLogger("train")

    log.info(f"Loading config from {args.config}")
    cfg = load_config(args.config)

    # Apply CLI overrides
    train_cfg  = cfg.get("training", {})
    env_cfg    = cfg.get("environment", {})
    sp_cfg     = cfg.get("self_play", {})

    total_ts   = args.total_timesteps or train_cfg.get("total_timesteps", 2_000_000)
    seed       = args.seed            or train_cfg.get("seed", 42)

    ppo_cfg    = build_ppo_config(cfg, device_override=args.device)

    log.info(f"Device: {ppo_cfg.device}")
    log.info(f"Total timesteps: {total_ts:,}")
    log.info(f"Seed: {seed}")
    log.info(f"CTDE (centralized critic): {ppo_cfg.centralized}")

    trainer = MARLTrainer(
        cfg             = ppo_cfg,
        env_kwargs      = env_cfg,
        log_dir         = train_cfg.get("log_dir",          "runs"),
        checkpoint_dir  = train_cfg.get("checkpoint_dir",   "checkpoints"),
        eval_interval   = train_cfg.get("eval_interval",    50),
        save_interval   = train_cfg.get("save_interval",    100),
        n_eval_episodes = train_cfg.get("n_eval_episodes",  20),
        seed            = seed,
    )

    if args.resume and args.tag:
        log.info(f"Resuming from checkpoint tag={args.tag}")
        trainer.load_checkpoint(args.tag)

    # Optional self-play wrapper
    if sp_cfg.get("enabled", False):
        sp_manager = SelfPlayManager(
            strategy          = sp_cfg.get("strategy",          "random"),
            pool_size         = sp_cfg.get("pool_size",         20),
            snapshot_interval = sp_cfg.get("snapshot_interval", 50),
            elo_k             = sp_cfg.get("elo_k",             32.0),
        )
        log.info(f"Self-play enabled: strategy={sp_manager.strategy}")
        # trainer.self_play_manager = sp_manager  # attach if trainer supports it

    trainer.train(total_timesteps=total_ts)


if __name__ == "__main__":
    main()
