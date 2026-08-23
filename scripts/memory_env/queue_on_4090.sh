#!/usr/bin/env bash
# Wait for the BTC trainings to release the 4090, then train the 3 MemoryEnv
# R2-Dreamer models sequentially (2cue/3cue/4cue, size25M, 30M steps) and
# produce outputs/report/memoryenv_reward_per_cue.png. Runs for days; nohup it.
set -uo pipefail
cd /home/filippo/GitHub/Crusoe-Cogniland
export PATH=/home/filippo/miniconda3/bin:$PATH
Q=outputs/train_logs/memoryenv_queue.log
T=outputs/train_logs/memoryenv_train.log
mkdir -p outputs/train_logs
echo "[queue] $(date) waiting for BTC (dreamerv3/ppo bridge_tunnel) to free the GPU..." | tee -a "$Q"
while pgrep -f "dreamerv3_bridge_tunnel.py|train_ppo_bridge_tunnel.py" >/dev/null 2>&1; do sleep 120; done
echo "[queue] $(date) GPU free -> starting 3 MemoryEnv trainings (logs -> $T)" | tee -a "$Q"
for CUE in 2cue 3cue 4cue; do
  echo "[queue] $(date) === training memory_${CUE} ===" | tee -a "$Q"
  PYTHONPATH=src conda run -n r2dreamer python r2dreamer_model/train.py \
    env=memory env.task=memory_${CUE} model=size25M env.steps=30e6 \
    device=cuda:0 seed=0 logdir=r2dreamer_model/runs/memory_${CUE} >> "$T" 2>&1 \
    && echo "[queue] $(date) memory_${CUE} DONE" | tee -a "$Q" \
    || echo "[queue] $(date) memory_${CUE} FAILED (continuing)" | tee -a "$Q"
done
echo "[queue] $(date) evaluating + plotting per-cue reward..." | tee -a "$Q"
PYTHONPATH=src conda run -n r2dreamer python scripts/memory_env/eval_r2dreamer.py \
  --ckpt-2cue r2dreamer_model/runs/memory_2cue/latest.pt \
  --ckpt-3cue r2dreamer_model/runs/memory_3cue/latest.pt \
  --ckpt-4cue r2dreamer_model/runs/memory_4cue/latest.pt --device cuda:0 >> "$Q" 2>&1 \
  && echo "[queue] $(date) ALL DONE -> outputs/report/memoryenv_reward_per_cue.png" | tee -a "$Q" \
  || echo "[queue] $(date) eval FAILED" | tee -a "$Q"
