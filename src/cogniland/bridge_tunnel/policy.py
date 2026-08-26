"""The PPO+GRU recurrent policy for bridge_tunnel (both variants).

Defined once here (was previously duplicated in the trainer scripts and reused via
a sys.path/importlib hack). The trainers, eval/viz scripts, the activation-dataset
builder, and the standalone steering kit all import it from this module so there is
a single source of truth. ``num_actions`` / ``num_tile_classes`` / ``obs_encoding``
are inferred from a checkpoint's state-dict at load time, so old checkpoints load.

obs → (tile-embed | one-hot) CNN(+CoordConv) → MLP(+scalars) → GRU → (actor, critic).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .tiles import NUM_TILES


def layer_init(layer, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOGRUPolicy(nn.Module):
    """Tile-embed/one-hot minimap → CNN → MLP(+scalars) → GRU → (Categorical, value)."""

    def __init__(self, obs_space, num_actions: int = 6, gru_hidden: int = 128,
                 embed_dim: int = 256, tile_embed_dim: int = 16,
                 num_tile_classes: int = NUM_TILES, obs_encoding: str = "embed",
                 belief_classes: int = 0, recurrent: bool = True):
        super().__init__()
        V, _ = obs_space["minimap"].shape
        n_scalars = obs_space["scalars"].shape[0]
        self.view = V
        self.obs_encoding = obs_encoding
        self.num_tile_classes = num_tile_classes
        if obs_encoding == "onehot":
            self.tile_embed = None
            in_c = num_tile_classes + 2                  # + CoordConv row/col
        else:
            self.tile_embed = nn.Embedding(num_tile_classes, tile_embed_dim)
            nn.init.normal_(self.tile_embed.weight, std=0.5)
            in_c = tile_embed_dim + 2
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(in_c, 32, kernel_size=3, padding=0)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, padding=0)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, padding=0)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, in_c, V, V)).shape[1]
        self.embed = nn.Sequential(
            layer_init(nn.Linear(n_flat + n_scalars, embed_dim)), nn.ReLU(),
        )
        # recurrent=False gives a memoryless control of matched head width: the
        # GRU is replaced by a per-step linear+ReLU with no carried state, so the
        # only difference from the recurrent agent is the memory itself.
        self.recurrent = recurrent
        if recurrent:
            self.gru = nn.GRU(embed_dim, gru_hidden, batch_first=False)
            for name, p in self.gru.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(p, 1.0)
                elif "bias" in name:
                    nn.init.constant_(p, 0.0)
            self.ff = None
        else:
            self.gru = None
            self.ff = nn.Sequential(layer_init(nn.Linear(embed_dim, gru_hidden)), nn.ReLU())
        self.actor = layer_init(nn.Linear(gru_hidden, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(gru_hidden, 1), std=1.0)
        # auxiliary belief head: classifies the map category from the recurrent
        # state (rocky/balanced/lakes); trained with a CE aux loss that shapes
        # the GRU representation. belief_scalar = P(lakes) - P(rocky) ∈ [-1,1].
        self.belief_classes = belief_classes
        self.belief = (layer_init(nn.Linear(gru_hidden, belief_classes), std=0.01)
                       if belief_classes else None)
        self.gru_hidden = gru_hidden

    def _encode(self, obs):
        mm = obs["minimap"].long()
        B, V, _ = mm.shape
        if self.obs_encoding == "onehot":
            emb = torch.nn.functional.one_hot(mm, self.num_tile_classes).float()
        else:
            emb = self.tile_embed(mm)
        rr = torch.linspace(-1, 1, V, device=mm.device).view(1, V, 1).expand(B, V, V)
        cc = torch.linspace(-1, 1, V, device=mm.device).view(1, 1, V).expand(B, V, V)
        coords = torch.stack([rr, cc], dim=-1)
        x = torch.cat([emb, coords], dim=-1).permute(0, 3, 1, 2)
        feat = self.cnn(x)
        feat = torch.cat([feat, obs["scalars"].float()], dim=-1)
        return self.embed(feat)

    def _gru_forward(self, obs_seq, done_seq, hidden):
        any_key = next(iter(obs_seq))
        T, B = obs_seq[any_key].shape[:2]
        flat = {k: v.flatten(0, 1) for k, v in obs_seq.items()}
        feat = self._encode(flat).reshape(T, B, -1)
        if not self.recurrent:
            # no carried state; the returned hidden is a placeholder of the
            # right shape so callers need no special case
            out = self.ff(feat.reshape(T * B, -1)).reshape(T, B, -1)
            return out, hidden
        h = hidden
        outs = []
        for t in range(T):
            mask = (1.0 - done_seq[t].float()).view(1, B, 1)
            h = h * mask
            y, h = self.gru(feat[t:t + 1], h)
            outs.append(y)
        return torch.cat(outs, dim=0), h

    def _heads(self, x):
        return self.actor(x), self.critic(x).squeeze(-1)

    def get_action_and_value(self, obs, hidden, done, action=None):
        obs_seq = {k: v.unsqueeze(0) for k, v in obs.items()}
        gru_out, h_new = self._gru_forward(obs_seq, done.unsqueeze(0), hidden)
        x = gru_out.squeeze(0)
        logits, value = self._heads(x)
        cat = Categorical(logits=logits)
        if action is None:
            action = cat.sample()
        return action, cat.log_prob(action), cat.entropy(), value, h_new

    def evaluate(self, obs_seq, done_seq, hidden, actions):
        gru_out, _ = self._gru_forward(obs_seq, done_seq, hidden)
        T, B = gru_out.shape[:2]
        x = gru_out.reshape(T * B, -1)
        logits, value = self._heads(x)
        cat = Categorical(logits=logits)
        a = actions.reshape(T * B)
        belief = self.belief(x) if self.belief is not None else None   # (T*B, C)
        return (cat.log_prob(a).reshape(T, B), cat.entropy().reshape(T, B),
                value.reshape(T, B), belief)

    # ── checkpoint helper: rebuild from a saved {"policy","args"} dict ──
    @classmethod
    def from_checkpoint(cls, ckpt: dict, obs_space, device="cpu"):
        sd = ckpt["policy"]; a = ckpt.get("args", {})
        if "tile_embed.weight" in sd:
            n_tiles = int(sd["tile_embed.weight"].shape[0]); enc = a.get("obs_encoding", "embed")
        else:
            n_tiles = int(sd["cnn.0.weight"].shape[1]) - 2; enc = "onehot"
        n_act = int(sd["actor.weight"].shape[0])
        recurrent = "gru.weight_ih_l0" in sd
        bc = int(sd["belief.weight"].shape[0]) if "belief.weight" in sd else 0
        pol = cls(obs_space, num_actions=n_act, gru_hidden=a.get("gru_hidden", 128),
                  embed_dim=a.get("embed_dim", 256), num_tile_classes=n_tiles,
                  obs_encoding=enc, belief_classes=bc, recurrent=recurrent).to(device)
        pol.load_state_dict(sd); pol.eval()
        return pol


__all__ = ["PPOGRUPolicy", "layer_init"]
