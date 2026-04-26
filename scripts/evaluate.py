#!/usr/bin/env python3
"""
evaluate.py — Load trained checkpoints and measure win rates.

Usage:
    python scripts/evaluate.py --tag final
    python scripts/evaluate.py --tag 500 --n_episodes 200 --config configs/default.yaml
"""

import argparse
import logging
import sys
import os
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo_agent import PPOConfig
from training.trainer import MARLTrainer


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained MARL Agents")
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--tag",        required=True, help="Checkpoint tag (e.g. 'final' or '500')")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    log = logging.getLogger("evaluate")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg   = cfg.get("environment", {})
    train_cfg = cfg.get("training", {})
    ppo_cfg   = PPOConfig(device=args.device)

    trainer = MARLTrainer(
        cfg            = ppo_cfg,
        env_kwargs     = env_cfg,
        checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints"),
        log_dir        = train_cfg.get("log_dir",        "runs"),
    )

    log.info(f"Loading checkpoints: tag={args.tag}")
    trainer.load_checkpoint(args.tag)

    log.info(f"Evaluating over {args.n_episodes} episodes …")
    win_rates = trainer.evaluate(n_episodes=args.n_episodes)

    print("\n" + "═" * 40)
    print("  Win Rates")
    print("═" * 40)
    for agent_id, wr in win_rates.items():
        bar = "█" * int(wr * 30)
        print(f"  {agent_id:<12} {wr:6.1%}  {bar}")
    print("═" * 40)

    # Self-play 1v1
    agents = list(trainer.possible_agents)
    if len(agents) >= 2:
        sp_wr = trainer.self_play_win_rate(agents[0], agents[1], n_episodes=args.n_episodes)
        print(f"\n  Self-play win rate ({agents[0]} vs {agents[1]}): {sp_wr:.1%}")


if __name__ == "__main__":
    main()
