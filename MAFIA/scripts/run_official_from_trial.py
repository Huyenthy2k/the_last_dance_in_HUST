#!/usr/bin/env python3
"""
Fork a best-valid checkpoint from an existing trial and continue training (or restart counting)
for a chosen number of epochs, reusing weights + replay buffer.

Example:
  python scripts/run_official_from_trial.py \
    --trial-dir res/RLcontroller/TD3/VNINDEX-10/2025-11-27-16-35-29-trial0 \
    --tag official-epoch1 \
    --extra-epochs 100 \
    --reset-counters   # start epoch numbering from 0 but use the best checkpoint weights/buffer

This will:
  - Copy the trial's checkpoint_best_valid into a new run folder named <tag>
  - Repoint checkpoint_info.json paths to the new location
  - Run RLcontroller for the requested epochs using the replay buffer + weights
"""

import argparse
import datetime
import json
import os
import shutil
import sys
from typing import Tuple

# Ensure repo root on path when executed from anywhere
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from config import Config
from entrance import RLcontroller


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fork a trial checkpoint and continue training")
    p.add_argument(
        "--trial-dir",
        required=True,
        help="Path to trial run directory (e.g., res/RLcontroller/TD3/VNINDEX-10/2025-11-27-16-35-29-trial0)",
    )
    p.add_argument(
        "--checkpoint-name",
        default="checkpoint_best_valid",
        help="Checkpoint subfolder to use (default: checkpoint_best_valid)",
    )
    p.add_argument(
        "--tag",
        default="official-epoch1",
        help="Name for the forked run directory (e.g., official-epoch1)",
    )
    p.add_argument(
        "--extra-epochs",
        type=int,
        default=1,
        help="Epochs to train. With --reset-counters (default), this is the total epochs to run. "
             "With --keep-counters, this is added on top of the checkpoint epoch.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=2022,
        help="Seed to use for the forked run",
    )
    reset_group = p.add_mutually_exclusive_group()
    reset_group.add_argument(
        "--reset-counters",
        dest="reset_counters",
        action="store_true",
        default=True,
        help="Reset epoch/day/timesteps in checkpoint_info to 0 so training starts counting from scratch (weights/buffer reused).",
    )
    reset_group.add_argument(
        "--keep-counters",
        dest="reset_counters",
        action="store_false",
        help="Keep epoch/day/timesteps from checkpoint_info to continue counting from that point.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only prepare the forked checkpoint/run directory; do not launch training",
    )
    return p.parse_args()


def copy_checkpoint(trial_dir: str, checkpoint_name: str, tag: str, reset_counters: bool) -> Tuple[str, dict]:
    trial_dir = os.path.abspath(trial_dir)
    base_dir = os.path.dirname(trial_dir)  # e.g., .../VNINDEX-10
    dest_run_dir = os.path.join(base_dir, tag)

    src_ckpt_dir = os.path.join(trial_dir, "checkpoints", checkpoint_name)
    if not os.path.exists(src_ckpt_dir):
        raise FileNotFoundError(f"Checkpoint folder not found: {src_ckpt_dir}")

    dest_ckpt_dir = os.path.join(dest_run_dir, "checkpoints", checkpoint_name)
    os.makedirs(os.path.dirname(dest_ckpt_dir), exist_ok=True)
    if os.path.exists(dest_ckpt_dir):
        shutil.rmtree(dest_ckpt_dir)
    shutil.copytree(src_ckpt_dir, dest_ckpt_dir)

    info_path = os.path.join(dest_ckpt_dir, "checkpoint_info.json")
    with open(info_path, "r") as f:
        info = json.load(f)

    source_info_path = os.path.join(src_ckpt_dir, "checkpoint_info.json")
    # Repoint file paths to the new checkpoint directory
    for key in [
        "rl_model_path",
        "mafia_observer_path",
        "replay_buffer_path",
        "env_state_path",
        "rng_state_path",
    ]:
        if key in info and info[key]:
            info[key] = os.path.join(dest_ckpt_dir, os.path.basename(info[key]))

    info["type"] = f"{info.get('type', 'epoch_best_valid')}_fork"
    info["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info["source_checkpoint"] = source_info_path
    if reset_counters:
        info["epoch"] = 0
        info["day_in_epoch"] = 0
        info["timesteps"] = 0
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # Ensure other run folders exist
    for sub in ["graph", "model"]:
        os.makedirs(os.path.join(dest_run_dir, sub), exist_ok=True)

    print(f"[PREP] Forked checkpoint -> {dest_ckpt_dir}")
    return info_path, info


def main():
    args = parse_args()
    ckpt_info_path, info = copy_checkpoint(
        trial_dir=args.trial_dir,
        checkpoint_name=args.checkpoint_name,
        tag=args.tag,
        reset_counters=args.reset_counters,
    )

    if args.dry_run:
        print("[DRY-RUN] Skipping training launch.")
        return

    start_epoch = info.get("epoch", 0)
    if args.reset_counters:
        target_epochs = args.extra_epochs
        print(f"[RUN] Reset counters: training for {target_epochs} epochs starting from epoch 0")
    else:
        target_epochs = start_epoch + args.extra_epochs
        print(f"[RUN] Continuing counts: start epoch {start_epoch} -> train to {target_epochs} (extra {args.extra_epochs})")
    print(f"[RUN] Using checkpoint info: {ckpt_info_path}")

    cfg = Config(seed_num=args.seed, current_date=args.tag)
    cfg.resume_from_checkpoint = ckpt_info_path
    cfg.auto_resume_from_latest = False
    cfg.reward_debug_steps = 0
    cfg.num_epochs = target_epochs

    RLcontroller(cfg)


if __name__ == "__main__":
    main()
