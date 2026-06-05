#!/usr/bin/env bash
# Stop all local sweep agents and their PPO trainer children.
pkill -f "wandb agent" 2>/dev/null || true
pkill -f "train_ppo_gru.py" 2>/dev/null || true
sleep 2
echo "stopped sweep agents + trainers"
