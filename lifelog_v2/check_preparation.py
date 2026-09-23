"""Run feature/split/weight/API checks without fitting any classifier."""
from pathlib import Path
import inspect
import json

import numpy as np
import pandas as pd

from . import feature_engineering as fe
from .data import load_dataset
from .experiments import subject_folds, fold_table, ablation_specs, suggest_spec, Config
from .metrics import select_policy, measure
from .modeling import subject_weights, make_estimator, FoldProcessor, Spec


def check_preparation(output="results_v2"):
    import optuna
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    _, paths = fe.resolve_paths()
    train = load_dataset(paths["train"], root / "cache")
    val = load_dataset(paths["val"], root / "cache")
    assert not set(train.subjects).intersection(val.subjects)
    folds = subject_folds(train.labels, 5, 2026)
    fold_table(train.labels, folds).to_csv(root / "prepared_development_folds.csv", index=False)
    outer = subject_folds(train.labels, 5, 9327)
    for i, (tr, te) in enumerate(outer):
        inner = subject_folds(train.labels.loc[tr], 3, 3026 + i)
        assert all(not set(a).intersection(te) and not set(b).intersection(te) for a, b in inner)
    rows = []
    for split, dataset in [("train", train), ("validation", val)]:
        for name, rep in dataset.representations.items():
            assert set(rep.groups) == set(dataset.subjects)
            assert not rep.meta.index.has_duplicates
            assert rep.X.columns.equals(rep.meta.index)
            assert not np.isinf(rep.X.to_numpy()).any()
            for tr, te in folds if split == "train" else []:
                assert not set(rep.groups[rep.rows_for(tr)]).intersection(rep.groups[rep.rows_for(te)])
            y = dataset.labels.reindex(rep.groups).to_numpy()
            weights = subject_weights(rep.groups, y, 1.)
            sums = pd.Series(weights).groupby(y).sum().to_numpy()
            assert np.allclose(sums, sums[0]), "Balanced classes must have equal total training weight."
            person_weight = pd.Series(weights, index=rep.groups).groupby(level=0).sum()
            for k in range(3):
                within = person_weight.loc[dataset.labels[dataset.labels == k].index]
                assert np.allclose(within, within.iloc[0])
            rows.append({"split": split, "representation": name, "subjects": len(dataset.labels),
                         "rows": len(rep.X), "features": rep.X.shape[1],
                         "missing_fraction": float(rep.X.isna().to_numpy().mean()),
                         "constant_or_empty_columns": int(rep.X.nunique().le(1).sum()),
                         "duplicate_complete_rows": int(rep.X.duplicated().sum())})
    # Preprocessing-only train/held-out isolation check. No classifier.fit calls.
    tr, te = folds[0]
    rep = train.representations["subject_plus"]
    subset = rep.X.loc[tr]
    processor = FoldProcessor(Spec(name="preprocessing_check", model="logreg", top_k=50), 2026)
    processor.fit(subset, train.labels.loc[tr].to_numpy(), tr, rep.meta)
    before = processor.medians_.copy()
    transformed = processor.transform(rep.X.loc[te])
    assert transformed.shape == (len(te), 50) and np.isfinite(transformed).all()
    extreme = rep.X.loc[te].copy() * 1e6
    processor.transform(extreme)
    assert processor.medians_.equals(before), "Held-out transforms must not update population statistics."
    # API validation: construct every ablation estimator and inspect weight routing.
    for spec in ablation_specs():
        estimator = make_estimator(spec, 2026, 1, binary=spec.model.startswith("hier_"))
        if spec.model != "lda":
            assert "sample_weight" in inspect.signature(estimator.fit).parameters
    # Generate search configurations only; no trials are evaluated or trained.
    sampler = optuna.samplers.RandomSampler(seed=2026)
    study = optuna.create_study(sampler=sampler)
    for _ in range(50):
        trial = study.ask()
        spec = suggest_spec(trial)
        assert not (spec.model == "lda" and spec.representation.startswith("window"))
        estimator = make_estimator(spec, 2026, 1, binary=spec.model.startswith("hier_"))
        estimator.get_params()
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    y = np.array([0, 0, 1, 1, 2, 2])
    scores = np.array([[.6, .3, .1], [.4, .5, .1], [.3, .6, .1], [.5, .4, .1], [.1, .2, .7], [.2, .3, .5]])
    offsets, tuned = select_policy(y, scores)
    assert tuned["macro_recall"] >= measure(y, scores)["macro_recall"]
    assert tuned["roc_auc_macro_ovr"] == measure(y, scores)["roc_auc_macro_ovr"]
    report = {"status": "passed", "classifier_fits": 0, "subject_overlap": 0,
              "fixed_ablation_conditions": len(ablation_specs()), "random_search_specs_constructed": 50,
              "checks": ["allowlisted features", "representation subject coverage", "outer/inner disjointness",
                         "class and subject sample weights", "train-only preprocessing", "estimator APIs",
                         "decision offsets do not change ROC inputs"], "features": rows}
    (root / "preparation_checks.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(rows).to_csv(root / "prepared_feature_audit.csv", index=False)
    train.representations["subject_plus"].meta.to_csv(root / "feature_dictionary.csv", index_label="feature")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    check_preparation()
