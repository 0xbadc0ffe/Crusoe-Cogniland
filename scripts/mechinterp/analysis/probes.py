"""Linear probes for belief (map category) and skill (committed action).

Probes are evaluated with a **grouped split over map_id** so a probe cannot cheat
by memorising map identity — train maps and test maps are disjoint, which is the
honest test of "is belief decodable from the activation". The same module fits:

  * categorical probe  — multinomial logistic regression -> P(class) + argmax pred
  * ordinal probe      — ridge regression onto the belief axis in [-1, 1]

Each probe also exposes its weight vector(s) as directions in *raw* activation
space (un-standardised), so they can be compared geometrically against the
difference-of-means directions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class CategoricalProbe:
    target: str                 # 'category' | 'final_commit' | ...
    classes: list
    accuracy: float
    balanced_accuracy: float
    confusion: np.ndarray       # rows=true, cols=pred (test split)
    n_train: int
    n_test: int
    pipe: Pipeline = field(repr=False)

    def proba(self, X) -> np.ndarray:
        return self.pipe.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        return self.pipe.predict(X)

    def directions(self) -> dict:
        """class -> weight vector in RAW activation space (one-vs-rest direction)."""
        clf = self.pipe.named_steps["clf"]
        scale = self.pipe.named_steps["scaler"].scale_
        coef = clf.coef_
        if coef.shape[0] == 1:               # binary -> two opposed dirs
            w = coef[0] / scale
            return {self.classes[1]: w, self.classes[0]: -w}
        return {c: coef[i] / scale for i, c in enumerate(clf.classes_)}


@dataclass
class OrdinalProbe:
    target: str
    r2: float
    spearman: float
    weight: np.ndarray          # direction in RAW activation space
    n_train: int
    n_test: int
    pipe: Pipeline = field(repr=False)

    def predict(self, X) -> np.ndarray:
        return np.clip(self.pipe.predict(X), -1.0, 1.0)


def _split(groups, n, test_frac, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    return next(gss.split(np.zeros(n), groups=groups))


def fit_categorical(X, y, groups, *, classes=None, C=1.0, max_iter=2000,
                    test_frac=0.25, seed=0) -> CategoricalProbe:
    y = np.asarray(y)
    tr, te = _split(groups, len(y), test_frac, seed)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=max_iter)),
    ])
    pipe.fit(X[tr], y[tr])
    pred = pipe.predict(X[te])
    classes = classes or sorted(np.unique(y).tolist())
    return CategoricalProbe(
        target="", classes=classes,
        accuracy=float((pred == y[te]).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y[te], pred)),
        confusion=confusion_matrix(y[te], pred, labels=classes),
        n_train=len(tr), n_test=len(te), pipe=pipe,
    )


def fit_ordinal(X, y, groups, *, test_frac=0.25, seed=0, alpha=1.0) -> OrdinalProbe:
    from scipy.stats import spearmanr
    y = np.asarray(y, dtype=float)
    tr, te = _split(groups, len(y), test_frac, seed)
    pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=alpha))])
    pipe.fit(X[tr], y[tr])
    pred = pipe.predict(X[te])
    w = pipe.named_steps["reg"].coef_ / pipe.named_steps["scaler"].scale_
    return OrdinalProbe(
        target="", r2=float(r2_score(y[te], pred)),
        spearman=float(spearmanr(y[te], pred).statistic),
        weight=w, n_train=len(tr), n_test=len(te), pipe=pipe,
    )


def proba_columns(probe: CategoricalProbe, X, prefix: str) -> dict:
    """Build a dict of dataframe columns: P(class) + argmax pred + confidence."""
    P = probe.proba(X)
    cols = {f"{prefix}_p_{c}": P[:, i] for i, c in enumerate(probe.pipe.named_steps["clf"].classes_)}
    cols[f"{prefix}_pred"] = probe.predict(X)
    cols[f"{prefix}_conf"] = P.max(1)
    return cols
