"""Linear probes, activation patching, and Representational Similarity Analysis."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import pdist, squareform


# ── Linear Probes ────────────────────────────────────────────────────────────

def train_linear_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
) -> dict[str, float]:
    """Train a logistic regression probe on frozen activations.

    Args:
        activations: [N, D] feature matrix.
        labels: [N] binary or multi-class labels.
        test_size: fraction for test split.
        random_state: for reproducibility.

    Returns:
        Dict with train_acc, test_acc, n_train, n_test.
    """
    # Flatten if needed
    if activations.ndim > 2:
        activations = activations.reshape(activations.shape[0], -1)

    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels, test_size=test_size,
        random_state=random_state, stratify=labels if len(np.unique(labels)) > 1 else None,
    )

    clf = LogisticRegression(
        max_iter=max_iter, random_state=random_state,
        solver="lbfgs", class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "coef": clf.coef_,
    }


def run_probes_for_concept(
    data_manager,
    concept_name: str,
    positive_flags: dict,
    negative_flags: dict,
    layers: list[str] = ("trunk_0", "trunk_2", "actor", "critic"),
) -> dict[str, dict[str, float]]:
    """Run linear probes for a binary concept across layers.

    Args:
        data_manager: TrajectoryDataManager instance.
        concept_name: for labeling results.
        positive_flags: kwargs for get_steps_where (positive class).
        negative_flags: kwargs for get_steps_where (negative class).
        layers: which activation layers to probe.

    Returns:
        Dict of layer_name → probe results dict.
    """
    pos_data = data_manager.get_steps_where(**positive_flags)
    neg_data = data_manager.get_steps_where(**negative_flags)

    results = {}
    for layer in layers:
        pos_acts = pos_data["activations"].get(layer)
        neg_acts = neg_data["activations"].get(layer)

        if pos_acts is None or neg_acts is None or len(pos_acts) == 0 or len(neg_acts) == 0:
            results[layer] = {"train_acc": 0.0, "test_acc": 0.0, "n_train": 0, "n_test": 0}
            continue

        # Flatten spatial dims
        if pos_acts.ndim > 2:
            pos_acts = pos_acts.reshape(pos_acts.shape[0], -1)
        if neg_acts.ndim > 2:
            neg_acts = neg_acts.reshape(neg_acts.shape[0], -1)

        X = np.concatenate([pos_acts, neg_acts])
        y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])

        # Subsample if too large
        if len(X) > 10000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), 10000, replace=False)
            X, y = X[idx], y[idx]

        try:
            results[layer] = train_linear_probe(X, y)
        except Exception as e:
            results[layer] = {"train_acc": 0.0, "test_acc": 0.0, "error": str(e)}

    return results


# ── Activation Contrast ──────────────────────────────────────────────────────

def compute_activation_contrast(
    data_manager,
    positive_flags: dict,
    negative_flags: dict,
    layer: str = "trunk_2",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean activation difference between two conditions.

    Returns:
        (mean_positive [D], mean_negative [D], delta [D])
    """
    pos_data = data_manager.get_steps_where(**positive_flags)
    neg_data = data_manager.get_steps_where(**negative_flags)

    pos_acts = pos_data["activations"].get(layer, np.empty((0, 0)))
    neg_acts = neg_data["activations"].get(layer, np.empty((0, 0)))

    if pos_acts.ndim > 2:
        pos_acts = pos_acts.reshape(pos_acts.shape[0], -1)
    if neg_acts.ndim > 2:
        neg_acts = neg_acts.reshape(neg_acts.shape[0], -1)

    mean_pos = pos_acts.mean(axis=0) if len(pos_acts) > 0 else np.zeros(1)
    mean_neg = neg_acts.mean(axis=0) if len(neg_acts) > 0 else np.zeros(1)
    delta = mean_pos - mean_neg

    return mean_pos, mean_neg, delta


# ── Activation Patching ──────────────────────────────────────────────────────

def activation_patching(
    model,
    obs_baseline: dict[str, torch.Tensor],
    obs_source: dict[str, torch.Tensor],
    patch_layer: str = "trunk.2",
    neuron_subset: list[int] | None = None,
) -> dict[str, float]:
    """Causal intervention: patch activations from source into baseline forward pass.

    Args:
        model: ActorCritic model.
        obs_baseline: observation dict for the baseline (target-not-visible).
        obs_source: observation dict for the source (target-visible).
        patch_layer: which layer to patch (dotted path, e.g., "trunk.2").
        neuron_subset: if provided, only patch these neuron indices.

    Returns:
        Dict with kl_divergence, baseline_action_probs, patched_action_probs.
    """
    from interpretability.collect import _resolve_submodule

    # Run source forward to capture its activations
    source_activation = {}

    def capture_hook(mod, inp, out):
        source_activation["act"] = out.detach().clone()

    submod = _resolve_submodule(model, patch_layer)
    h = submod.register_forward_hook(capture_hook)
    with torch.no_grad():
        feat_src = model._features(obs_source)
    h.remove()

    # Run baseline forward with patching hook
    source_act = source_activation["act"]

    def patch_hook(mod, inp, out):
        patched = out.clone()
        if neuron_subset is not None:
            patched[:, neuron_subset] = source_act[:, neuron_subset]
        else:
            patched[:] = source_act
        return patched

    h = submod.register_forward_hook(patch_hook)
    with torch.no_grad():
        feat_patched = model._features(obs_baseline)
        logits_patched = model.actor(feat_patched)
    h.remove()

    # Baseline (no patching)
    with torch.no_grad():
        feat_base = model._features(obs_baseline)
        logits_base = model.actor(feat_base)

    probs_base = torch.softmax(logits_base, dim=-1)
    probs_patched = torch.softmax(logits_patched, dim=-1)

    # KL divergence
    kl = (probs_patched * (probs_patched.log() - probs_base.log())).sum(dim=-1).mean().item()

    return {
        "kl_divergence": kl,
        "baseline_probs": probs_base[0].cpu().numpy(),
        "patched_probs": probs_patched[0].cpu().numpy(),
    }


def batch_activation_patching_by_neuron(
    model,
    obs_baseline: dict[str, torch.Tensor],
    obs_source: dict[str, torch.Tensor],
    patch_layer: str = "trunk.2",
    top_k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Patch individual neurons and measure KL divergence for each.

    Returns:
        (neuron_indices [top_k], kl_values [top_k]) sorted by importance.
    """
    from interpretability.collect import _resolve_submodule

    # Capture source activations
    source_act_cache = {}

    def capture(mod, inp, out):
        source_act_cache["act"] = out.detach().clone()

    submod = _resolve_submodule(model, patch_layer)
    h = submod.register_forward_hook(capture)
    with torch.no_grad():
        model._features(obs_source)
    h.remove()
    source_act = source_act_cache["act"]
    n_neurons = source_act.shape[-1]

    # Baseline logits
    with torch.no_grad():
        feat_base = model._features(obs_baseline)
        logits_base = model.actor(feat_base)
    probs_base = torch.softmax(logits_base, dim=-1)

    kl_values = np.zeros(n_neurons)

    for i in range(n_neurons):
        def _patch_single(mod, inp, out, idx=i):
            patched = out.clone()
            patched[:, idx] = source_act[:, idx]
            return patched

        h = submod.register_forward_hook(_patch_single)
        with torch.no_grad():
            feat_p = model._features(obs_baseline)
            logits_p = model.actor(feat_p)
        h.remove()

        probs_p = torch.softmax(logits_p, dim=-1)
        kl = (probs_p * (probs_p.log() - probs_base.log())).sum(dim=-1).mean().item()
        kl_values[i] = kl

    top_idx = np.argsort(kl_values)[::-1][:top_k]
    return top_idx, kl_values[top_idx]


# ── Representational Similarity Analysis (RSA) ──────────────────────────────

def compute_rdm(
    activations_by_group: dict[str, np.ndarray],
    metric: str = "cosine",
) -> tuple[np.ndarray, list[str]]:
    """Compute representational dissimilarity matrix.

    Args:
        activations_by_group: group_name → [N_g, D] mean activations.
        metric: distance metric for pdist.

    Returns:
        (rdm [G, G], group_names [G])
    """
    group_names = sorted(activations_by_group.keys())
    means = []
    for name in group_names:
        acts = activations_by_group[name]
        if acts.ndim > 2:
            acts = acts.reshape(acts.shape[0], -1)
        means.append(acts.mean(axis=0))

    mean_matrix = np.stack(means)  # [G, D]
    distances = squareform(pdist(mean_matrix, metric=metric))
    return distances, group_names


def compute_terrain_rdms(
    data_manager,
    layers: list[str] = ("cnn_0", "cnn_5", "trunk_0", "trunk_2"),
    terrain_names: tuple[str, ...] = (
        "ocean", "deep_water", "water", "beach", "sandy",
        "grassland", "forest", "rocky", "mountains",
    ),
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Compute terrain-based RDM for each layer.

    Groups steps by terrain type, computes mean activation per group,
    then pairwise dissimilarity.

    Returns:
        Dict of layer_name → (rdm [9, 9], terrain_labels [9]).
    """
    results = {}

    for layer in layers:
        acts_flat, _, terrain_flat = data_manager.get_all_activations_flat(layer)
        if len(acts_flat) == 0:
            continue

        groups = {}
        for t_idx, t_name in enumerate(terrain_names):
            mask = terrain_flat.astype(int) == t_idx
            if mask.any():
                groups[t_name] = acts_flat[mask]

        if len(groups) >= 2:
            rdm, names = compute_rdm(groups)
            results[layer] = (rdm, names)

    return results


# ── LDA for cluster separation ───────────────────────────────────────────────

def cluster_lda(
    activations: np.ndarray,
    cluster_labels: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, LinearDiscriminantAnalysis]:
    """Project activations using LDA to best separate clusters.

    Returns:
        (projected [N, n_components], fitted LDA model)
    """
    # Filter out noise label (-1 from HDBSCAN)
    valid = cluster_labels >= 0
    if valid.sum() < 10:
        return np.zeros((len(activations), n_components)), LinearDiscriminantAnalysis()

    X = activations[valid]
    y = cluster_labels[valid]

    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    n_classes = len(np.unique(y))
    n_comp = min(n_components, n_classes - 1, X.shape[1])
    if n_comp < 1:
        return np.zeros((len(activations), n_components)), LinearDiscriminantAnalysis()

    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    lda.fit(X, y)

    # Project all points (including noise)
    X_all = activations.reshape(activations.shape[0], -1) if activations.ndim > 2 else activations
    projected = lda.transform(X_all)

    # Pad to requested n_components if needed
    if projected.shape[1] < n_components:
        projected = np.pad(projected, ((0, 0), (0, n_components - projected.shape[1])))

    return projected, lda
