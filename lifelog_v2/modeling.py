"""Fold-local preprocessing, subject-weighted learning and subject predictions."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import warnings

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from catboost import CatBoostClassifier

from .data import Representation


@dataclass
class Spec:
    name: str
    parent: str | None = None
    representation: str = "subject_plus"
    view: str = "all"
    model: str = "catboost"
    balance: float = 1.0
    top_k: int | str = 100
    ranking: str = "anova"
    correlation_limit: float = 1.0
    pca: int = 0
    pooling: str = "mean"
    params: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def normalize_scores(values):
    p = np.asarray(values, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("Expected finite N x 3 model scores in CN/MCI/DEM order.")
    p = np.clip(p, 1e-12, None)
    return p / p.sum(axis=1, keepdims=True)


def allowed_columns(meta, view):
    if view == "all":
        mask = np.ones(len(meta), dtype=bool)
    elif view == "physiology":
        mask = meta.family.eq("physiology")
    elif view == "no_quality":
        mask = ~meta.family.eq("quality")
    elif view == "no_vendor":
        mask = ~meta.family.eq("vendor")
    elif view == "no_coupling":
        mask = ~meta.modality.eq("coupling")
    elif view in {"activity", "sleep"}:
        mask = meta.modality.eq(view)
    else:
        raise ValueError(view)
    return meta.index[mask].tolist()


class FoldProcessor:
    """Fit population statistics and ranking on TRAIN subjects only.

    Window models use one mean feature vector per training subject for ranking,
    imputation and scaling so long recordings do not dominate preprocessing.
    """

    def __init__(self, spec, seed):
        self.spec, self.seed = spec, seed

    def fit(self, X, y, groups, meta):
        candidate = X.loc[:, allowed_columns(meta, self.spec.view)].copy()
        candidate = candidate.replace([np.inf, -np.inf], np.nan)
        centers = candidate.groupby(np.asarray(groups), sort=True).mean()
        labels = pd.Series(y, index=groups).groupby(level=0).first().reindex(centers.index)
        n_labels = pd.Series(y, index=groups).groupby(level=0).nunique()
        assert n_labels.eq(1).all(), "Conflicting labels within a subject."
        usable = centers.notna().sum().ge(max(2, int(np.ceil(len(centers) * .1)))) & centers.nunique().gt(1)
        centers = centers.loc[:, usable]
        if not centers.shape[1]:
            raise ValueError("No usable features in this training fold.")
        medians = centers.median().fillna(0)
        imputed = centers.fillna(medians)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if self.spec.ranking == "mutual_info":
                scores = mutual_info_classif(imputed, labels, random_state=self.seed)
            else:
                scores, _ = f_classif(imputed, labels)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=1e30)
        ranked = imputed.columns[np.argsort(-scores, kind="stable")].tolist()
        limit = len(ranked) if self.spec.top_k == "all" else min(int(self.spec.top_k), len(ranked))
        selected = []
        if self.spec.correlation_limit < 1.0:
            correlations = imputed.corr().abs().fillna(0)
            for c in ranked:
                if not selected or correlations.loc[c, selected].max() < self.spec.correlation_limit:
                    selected.append(c)
                if len(selected) >= limit:
                    break
        else:
            selected = ranked[:limit]
        self.columns_ = selected
        self.medians_ = medians.loc[selected]
        self.center_ = imputed[selected].median()
        self.scale_ = (imputed[selected].quantile(.75) - imputed[selected].quantile(.25)).replace(0, 1)
        self.scaled_ = self.spec.model in {"logreg", "svm", "lda", "gaussian_nb", "hier_logreg"}
        self.native_missing_ = self.spec.model in {"catboost", "hier_catboost", "histgb"}
        self.pca_ = None
        if self.spec.pca:
            if not self.scaled_:
                raise ValueError("PCA is restricted to scaled model families.")
            matrix = self._numeric(centers)
            count = min(int(self.spec.pca), matrix.shape[1], matrix.shape[0] - 1)
            self.pca_ = PCA(n_components=count, svd_solver="full", random_state=self.seed).fit(matrix)
        return self

    def _numeric(self, X):
        data = X.reindex(columns=self.columns_).replace([np.inf, -np.inf], np.nan)
        if not self.native_missing_:
            data = data.fillna(self.medians_)
        if self.scaled_:
            data = ((data - self.center_) / self.scale_).clip(-20, 20)
        return data.to_numpy(dtype=float)

    def transform(self, X):
        absent = set(self.columns_) - set(X.columns)
        if absent:
            raise ValueError(f"Missing model features: {sorted(absent)[:5]}")
        values = self._numeric(X)
        return self.pca_.transform(values) if self.pca_ is not None else values


def subject_weights(groups, y, balance):
    frame = pd.DataFrame({"group": groups, "y": y})
    assert frame.groupby("group").y.nunique().eq(1).all()
    people = frame.drop_duplicates("group")
    class_counts = people.y.value_counts()
    observation_counts = frame.group.value_counts()
    # Every person contributes equally before optional class balancing.
    weights = np.array([(1 / observation_counts[g]) * class_counts[int(label)] ** (-balance)
                        for g, label in zip(groups, y)], dtype=float)
    return weights / weights.mean()


def make_estimator(spec, seed, threads, binary=False):
    p = dict(spec.params)
    kind = spec.model.removeprefix("hier_")
    if kind == "catboost":
        defaults = dict(iterations=600, depth=4, learning_rate=.035, l2_leaf_reg=10,
                        random_strength=.5, boosting_type="Ordered", border_count=64)
        defaults.update(p)
        return CatBoostClassifier(**defaults, loss_function="Logloss" if binary else "MultiClass",
                                  thread_count=threads, random_seed=seed, verbose=False,
                                  allow_writing_files=False, task_type="CPU")
    if kind == "logreg":
        return LogisticRegression(C=float(p.get("C", .1)), solver="lbfgs", max_iter=4000)
    if kind == "svm":
        return SVC(C=float(p.get("C", 1.0)), gamma=p.get("gamma", "scale"),
                   kernel=p.get("kernel", "rbf"), probability=False, decision_function_shape="ovr",
                   cache_size=512, random_state=seed)
    if kind in {"extratrees", "randomforest"}:
        cls = ExtraTreesClassifier if kind == "extratrees" else RandomForestClassifier
        defaults = dict(n_estimators=500, min_samples_leaf=2, max_features="sqrt", max_depth=None)
        defaults.update(p)
        return cls(**defaults, n_jobs=threads, random_state=seed)
    if kind == "histgb":
        defaults = dict(max_iter=300, learning_rate=.05, max_leaf_nodes=7,
                        min_samples_leaf=10, l2_regularization=5.0)
        defaults.update(p)
        # No internal random row split, especially for repeated windows.
        return HistGradientBoostingClassifier(**defaults, early_stopping=False, random_state=seed)
    if kind == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=p.get("shrinkage", "auto"))
    if kind == "gaussian_nb":
        return GaussianNB(var_smoothing=p.get("var_smoothing", 1e-3))
    raise ValueError(kind)


def estimator_scores(estimator, values):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(values)
    # SVM scores are normalized decision scores, NOT calibrated risk estimates.
    decision = estimator.decision_function(values)
    if decision.ndim == 1:
        decision = np.column_stack([-decision, decision])
    return softmax(decision, axis=1)


class HierarchicalModel:
    """P(CN), P(MCI), P(DEM) from impairment and conditional DEM stages."""

    def __init__(self, spec, seed, threads):
        self.spec, self.seed, self.threads = spec, seed, threads

    def fit(self, X, y, groups):
        self.first_ = make_estimator(self.spec, self.seed, self.threads, binary=True)
        self.second_ = make_estimator(self.spec, self.seed + 1, self.threads, binary=True)
        impaired = y > 0
        first_y = impaired.astype(int)
        second_y = (y[impaired] == 2).astype(int)
        self.first_.fit(X, first_y, sample_weight=subject_weights(groups, first_y, self.spec.balance))
        self.second_.fit(X[impaired], second_y,
                         sample_weight=subject_weights(groups[impaired], second_y, self.spec.balance))
        return self

    def predict_proba(self, X):
        impairment = estimator_scores(self.first_, X)[:, 1]
        dem = estimator_scores(self.second_, X)[:, 1]
        return np.column_stack([1 - impairment, impairment * (1 - dem), impairment * dem])


@dataclass
class FittedModel:
    spec: Spec
    processor: FoldProcessor
    estimator: object
    training_subjects: tuple
    seed: int

    def predict(self, rep, subjects):
        rows = rep.rows_for(subjects)
        row_scores = normalize_scores(estimator_scores(self.estimator, self.processor.transform(rep.X.iloc[rows])))
        values = np.log(row_scores) if self.spec.pooling == "geometric" else row_scores
        pooled = pd.DataFrame(values, index=rep.groups[rows]).groupby(level=0).mean().reindex(subjects)
        if pooled.isna().any().any():
            raise ValueError("A requested subject has no predictions.")
        return normalize_scores(np.exp(pooled.to_numpy()) if self.spec.pooling == "geometric" else pooled.to_numpy())


def fit_model(rep: Representation, labels, subjects, spec, *, seed, threads):
    rows = rep.rows_for(subjects)
    groups = rep.groups[rows]
    y = labels.reindex(groups).to_numpy(dtype=int)
    assert set(groups) == set(subjects)
    assert set(y) == {0, 1, 2}
    processor = FoldProcessor(spec, seed).fit(rep.X.iloc[rows], y, groups, rep.meta)
    values = processor.transform(rep.X.iloc[rows])
    if spec.model.startswith("hier_"):
        estimator = HierarchicalModel(spec, seed, threads).fit(values, y, groups)
    else:
        estimator = make_estimator(spec, seed, threads)
        if spec.model == "lda":
            if len(set(groups)) != len(groups):
                raise ValueError("LDA is subject-summary only; repeated-window weighting is unsupported.")
            counts = np.bincount(y, minlength=3).astype(float)
            priors = counts ** (1 - spec.balance)
            estimator.set_params(priors=priors / priors.sum())
            estimator.fit(values, y)
        else:
            estimator.fit(values, y, sample_weight=subject_weights(groups, y, spec.balance))
    return FittedModel(spec, processor, estimator, tuple(sorted(set(groups))), seed)


@dataclass
class EnsembleModel:
    models: list[FittedModel]
    weights: np.ndarray
    offsets: np.ndarray
    feature_config: dict
    provenance: dict

    def predict(self, representations, subjects):
        score = sum(float(w) * model.predict(representations[model.spec.representation], subjects)
                    for w, model in zip(self.weights, self.models))
        score = normalize_scores(score)
        prediction = (np.log(score) + self.offsets).argmax(axis=1)
        return score, prediction

    def predict_csv(self, activity_csv, sleep_csv):
        from . import feature_engineering as fe
        from .data import inference_representations
        if dict(fe.CFG) != self.feature_config:
            raise ValueError("Inference feature configuration differs from training configuration.")
        needed = sorted({int(m.spec.representation.removeprefix("window")) for m in self.models
                         if m.spec.representation.startswith("window")})
        reps, subjects = inference_representations(activity_csv, sleep_csv, windows=needed)
        score, pred = self.predict(reps, subjects)
        frame = pd.DataFrame(score, index=subjects, columns=["score_CN", "score_MCI", "score_DEM"])
        frame.index.name = "subject_hash"
        frame["predicted"] = np.array(["CN", "MCI", "DEM"])[pred]
        return frame
