#!/usr/bin/env python3
"""Compute test metrics for all sweep checkpoints and save to data/sweep_test_results.csv."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pandas as pd
from omegaconf import OmegaConf

from cogniland.env.types import EnvConfig
from cogniland.env.wrappers import BatchedIslandEnv
from cogniland.env.dataset import MapDataset
from cogniland.models import build_model
from cogniland.eval import CognilandSummarizer, EvalRunner

SWEEP_RUN_IDS = [
    "02agthx8","0fjc3yvj","0ot2ghk8","27xcjita","2np83te6","3psg1lr8",
    "3x0j9c69","4j0mo0ws","4yhibzyb","55po9bfo","6s3d2u1d","7nfloswx",
    "8sl8z1ma","96x12583","99v68sf7","9xqfrhqc","a948o8tq","ahk1o7q9",
    "bpkulemc","bw90jdtn","e2ijy7u4","e3c0itct","eyeuudwv","f665rxcq",
    "ie0scu8d","jzpgle7e","leo31vpb","lx9pzxdr","m9ssmoyp","q0cbeblk",
    "qd9b25p9","queyzb76","sbd3ct4q","v18v9a9d","vi60lv82","zwpxpm5l",
]

ARTIFACTS_DIR = Path("artifacts_sweep")
CACHE_PATH    = Path("data/sweep_test_results.csv")


def main():
    if CACHE_PATH.exists():
        print(f"Cache already exists at {CACHE_PATH} — delete it to recompute.")
        return

    device_str = ("cuda" if torch.cuda.is_available()
                  else ("mps" if torch.backends.mps.is_available() else "cpu"))
    device = torch.device(device_str)
    print(f"Device: {device_str}")

    # Load configs
    _env_yaml   = OmegaConf.load("configs/env/default.yaml")
    _model_yaml = OmegaConf.load("configs/models/ppo.yaml")
    cfg = OmegaConf.create({
        "device": device_str,
        "env": OmegaConf.to_container(_env_yaml, resolve=True),
        "models": OmegaConf.to_container(_model_yaml, resolve=True),
    })

    dataset    = MapDataset.load(cfg.models["training"]["dataset"]["path"])
    n_test_eps = len(dataset.test_maps)
    env_config = EnvConfig.from_hydra(cfg)
    summarizer = CognilandSummarizer()
    print(f"Test maps: {n_test_eps}")

    records = []
    missing = []
    for i, run_id in enumerate(SWEEP_RUN_IDS):
        ckpt_path = ARTIFACTS_DIR / run_id / "ckpt_best.pt"
        if not ckpt_path.exists():
            print(f"  [{i+1:2d}/{len(SWEEP_RUN_IDS)}] {run_id}: MISSING — skip")
            missing.append(run_id)
            continue

        eval_env = BatchedIslandEnv(
            env_config,
            num_envs=n_test_eps,
            world_maps=dataset.test_maps,
        )
        runner = EvalRunner(eval_env, env_config, device_str)

        model = build_model(cfg)
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.model.load_state_dict(ckpt["model_state_dict"])
        model.model.to(device)
        model.model.eval()

        result = runner.run(
            policy_fn=lambda obs: model.get_deterministic_action(obs),
            n_episodes=n_test_eps,
            mode="det",
            split="test",
            global_step=int(ckpt.get("step", 0)),
        )
        m = summarizer.scalar_metrics(result)

        records.append({
            "run_id":             run_id,
            "test_success_rate":  m["test_det/env/success_rate"],
            "test_directness":    m["test_det/env/directness_mean"],
            "test_exploration":   m["test_det/env/exploration_mean"],
            "test_risk_exposure": m["test_det/env/risk_exposure_mean"],
        })
        sr = m["test_det/env/success_rate"]
        print(f"  [{i+1:2d}/{len(SWEEP_RUN_IDS)}] {run_id}: success_rate={sr:.3f}")

    if missing:
        print(f"\nWarning: {len(missing)} checkpoints missing: {missing}")

    df = pd.DataFrame(records)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {CACHE_PATH}")
    print(df.describe())


if __name__ == "__main__":
    main()
