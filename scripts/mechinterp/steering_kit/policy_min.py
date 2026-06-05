"""Self-contained copy of the PPO+GRU policy — torch only, NO cogniland import.

Identical architecture to cogniland.bridge_tunnel.policy.PPOGRUPolicy, so a saved
``{"policy": state_dict, "args": {...}}`` checkpoint loads here unchanged.
``from_checkpoint`` infers num_actions / tile-count / obs_encoding from the
state-dict so both bt (5-scalar) and btc (7-scalar) checkpoints work.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

NUM_TILES = 9


def _li(layer, std=np.sqrt(2), b=0.0):
    nn.init.orthogonal_(layer.weight, std); nn.init.constant_(layer.bias, b); return layer


class PPOGRUPolicy(nn.Module):
    def __init__(self, view, n_scalars, num_actions=6, gru_hidden=128, embed_dim=256,
                 tile_embed_dim=16, num_tile_classes=NUM_TILES, obs_encoding="embed"):
        super().__init__()
        self.obs_encoding = obs_encoding
        self.num_tile_classes = num_tile_classes
        if obs_encoding == "onehot":
            self.tile_embed = None; in_c = num_tile_classes + 2
        else:
            self.tile_embed = nn.Embedding(num_tile_classes, tile_embed_dim); in_c = tile_embed_dim + 2
        self.cnn = nn.Sequential(
            _li(nn.Conv2d(in_c, 32, 3)), nn.ReLU(), _li(nn.Conv2d(32, 32, 3)), nn.ReLU(),
            _li(nn.Conv2d(32, 32, 3)), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, in_c, view, view)).shape[1]
        self.embed = nn.Sequential(_li(nn.Linear(n_flat + n_scalars, embed_dim)), nn.ReLU())
        self.gru = nn.GRU(embed_dim, gru_hidden, batch_first=False)
        self.actor = _li(nn.Linear(gru_hidden, num_actions), std=0.01)
        self.critic = _li(nn.Linear(gru_hidden, 1), std=1.0)
        self.gru_hidden = gru_hidden

    def _encode(self, obs):
        mm = obs["minimap"].long(); B, V, _ = mm.shape
        emb = (torch.nn.functional.one_hot(mm, self.num_tile_classes).float()
               if self.obs_encoding == "onehot" else self.tile_embed(mm))
        rr = torch.linspace(-1, 1, V).view(1, V, 1).expand(B, V, V)
        cc = torch.linspace(-1, 1, V).view(1, 1, V).expand(B, V, V)
        x = torch.cat([emb, torch.stack([rr, cc], -1)], -1).permute(0, 3, 1, 2)
        return self.embed(torch.cat([self.cnn(x), obs["scalars"].float()], -1))

    def step(self, obs, h, inject=None):
        """One step: returns (logits, value, gru_h). ``inject`` (1,1,H) is added to
        the GRU hidden (and persisted) — the activation-steering hook."""
        feat = self._encode(obs).reshape(1, 1, -1)
        y, h = self.gru(feat, h)
        if inject is not None:
            y = y + inject; h = h + inject
        x = y.squeeze(0)
        return self.actor(x), self.critic(x).squeeze(-1), h

    @classmethod
    def from_checkpoint(cls, ckpt, view, n_scalars, device="cpu"):
        sd = ckpt["policy"]; a = ckpt.get("args", {})
        if "tile_embed.weight" in sd:
            n_tiles = int(sd["tile_embed.weight"].shape[0]); enc = a.get("obs_encoding", "embed")
        else:
            n_tiles = int(sd["cnn.0.weight"].shape[1]) - 2; enc = "onehot"
        n_act = int(sd["actor.weight"].shape[0])
        p = cls(view, n_scalars, num_actions=n_act, gru_hidden=a.get("gru_hidden", 128),
                embed_dim=a.get("embed_dim", 256), num_tile_classes=n_tiles, obs_encoding=enc).to(device)
        p.load_state_dict(sd); p.eval()
        return p
