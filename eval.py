#!/usr/bin/env python3
"""Evaluation entry point — Fetches model checkpoint and original config from WandB by run ID.

Usage:
    python eval.py 3disj6tl
"""

import os
import sys
import torch
import wandb
from omegaconf import OmegaConf

def main():
    if len(sys.argv) < 2:
        print("Error: You must provide a wandb_run_id to evaluate.")
        print("Example: python eval.py YOUR_RUN_ID")
        return

    run_id_arg = sys.argv[1]
    # If the user accidentally types wandb_run_id=ID, extract just the ID
    if run_id_arg.startswith("wandb_run_id="):
        run_id = run_id_arg.split("=")[1]
    else:
        run_id = run_id_arg
    
    # Based on your previous artifacts, assuming standard Crusoe-Cogniland paths
    project_path = "crusoe/cogniland"
    
    print(f"Initializing WandB API to fetch config for run: {run_id}")
    api = wandb.Api()
    try:
        run_data = api.run(f"{project_path}/{run_id}")
    except Exception as e:
        print(f"Failed to find WandB run at {project_path}/{run_id}.")
        print(f"Error: {e}")
        return

    # Extract the exact Config from when it was trained
    print("Reconstructing original training configuration...")
    
    # WandB's run.config returns values in a flat structure; rebuild the dictionary
    raw_config = {k: v for k, v in run_data.config.items() if not k.startswith('_')}
    cfg = OmegaConf.create(raw_config)

    # Initialize a lightweight run to download artifacts
    entity = run_data.entity
    project = run_data.project
    run = wandb.init(
        project=project, 
        entity=entity,
        job_type="evaluation"
    )

    artifact_name = f"{cfg.models.name}_agent_{run_id}:latest"
    print(f"Fetching artifact: {artifact_name}...")
    
    try:
        # Request explicitly from the entity/project namespace
        artifact = run.use_artifact(f"{entity}/{project}/{artifact_name}")
    except wandb.errors.CommError as e:
        print("Error fetching artifact. Make sure the run ID saved a checkpoint.")
        print(e)
        return

    artifact_dir = artifact.download()
    
    ckpt_files = [f for f in os.listdir(artifact_dir) if f.endswith(".pt")]
    if not ckpt_files:
        print("No .pt files found in the downloaded artifact.")
        return
        
    ckpt_path = os.path.join(artifact_dir, ckpt_files[-1])
    print(f"Downloaded checkpoint to: {ckpt_path}")

    # Build the model using the historically accurate configuration
    print("Building model architecture from historical config...")
    from cogniland.models import build_model
    model = build_model(cfg)

    # Load the checkpoint
    print("Loading checkpoint weights...")
    
    # Handle device dynamically avoiding missing 'device' attributes
    if getattr(cfg, "device", "cpu") == "auto":
        device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device_str = getattr(cfg, "device", "cpu")
        
    device = torch.device(device_str)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model.model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("--- Evaluation Ready ---")
    print(f"Model ID: {run_id}")
    print(f"Model Architecture: {cfg.models.name}")
    print(f"Parameters: {param_count:,}")
    print("Weights loaded successfully.")

    # --- EVALUATION ---
    # By default, evaluating the model uses the same validation map it saw during
    # training (cfg.env.seed + 1000). To evaluate on fully unseen procedural maps
    # and test true zero-shot generalization, simply override the seed here:
    # cfg.env.seed = 9999
    
    print("Running evaluation metrics...")
    metrics = model._run_eval(cfg, logger=None, global_step=0)
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print("="*40)
    
    run.finish()

if __name__ == "__main__":
    main()
