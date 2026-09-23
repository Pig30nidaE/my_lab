"""Reproducible ablations, nested subject CV, OOF search and final fitting.

No learning runs on import. Call run_experiment explicitly when ready to train.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import platform
import sys
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from . import feature_engineering as fe
from .data import load_dataset
from .metrics import measure, objective, select_policy, intervals, target_status, NAMES
from .modeling import Spec, fit_model, EnsembleModel, normalize_scores


@dataclass
class Config:
    seed: int = 2026
    threads: int = 4
    development_folds: int = 5
    outer_folds: int = 5
    inner_folds: int = 3
    n_trials: int = 80
    nested_trials: int = 20
    run_nested: bool = True
    final_seeds: tuple = (2026, 2037, 2048)
    max_ensemble_members: int = 4
    ensemble_steps: int = 8
    tune_decisions: bool = True
    bootstrap_repeats: int = 2000
    threshold: float = .8
    cache_dir: str = "results_v2/cache"
    output_dir: str = "results_v2/run_full"


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def source_signature():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}


def versions():
    return {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "optuna", "joblib"]}


def subject_folds(labels, n_splits, seed):
    """Stratify the UNIQUE SUBJECT table, then map each person to all rows.

    This is a grouped split by construction, including window representations.
    Exact class stratification is possible because each person has one label.
    """
    assert labels.index.is_unique
    if labels.value_counts().min() < n_splits:
        raise ValueError("Too few minority-class subjects for the requested folds.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    seen = []
    subjects = labels.index.to_numpy()
    for train_idx, test_idx in splitter.split(subjects, labels.to_numpy()):
        tr, te = subjects[train_idx], subjects[test_idx]
        assert not set(tr).intersection(te)
        assert set(labels.loc[tr]) == set(labels.loc[te]) == {0, 1, 2}
        folds.append((tr, te))
        seen.extend(te)
    assert len(seen) == len(set(seen)) == len(labels)
    return folds


def fold_table(labels, folds):
    rows = []
    for i, (tr, te) in enumerate(folds, 1):
        row = {"fold": i, "train_subjects": len(tr), "test_subjects": len(te)}
        for k, name in enumerate(NAMES):
            row[f"train_{name}"] = int((labels.loc[tr] == k).sum())
            row[f"test_{name}"] = int((labels.loc[te] == k).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_specs():
    """Prespecified paired retraining comparisons; parent changes one factor."""
    baseline = Spec(name="A00_v1_reference", representation="subject_v1", top_k="all")
    specs = [baseline]
    for i, view in enumerate(["physiology", "no_quality", "no_vendor", "no_coupling", "activity", "sleep"], 1):
        specs.append(replace(baseline, name=f"A{i:02d}_{view}", parent=baseline.name, view=view))
    rich = replace(baseline, name="A07_enhanced", parent=baseline.name, representation="subject_plus")
    top100 = replace(rich, name="A08_top100", parent=rich.name, top_k=100)
    specs.extend([rich, top100, replace(rich, name="A09_top50", parent=rich.name, top_k=50),
                  replace(top100, name="A10_corr095", parent=top100.name, correlation_limit=.95)])
    specs.extend([replace(top100, name="A11_window14", parent=top100.name, representation="window14"),
                  replace(top100, name="A12_window28", parent=top100.name, representation="window28"),
                  replace(top100, name="A13_window14_geometric", parent="A11_window14", representation="window14", pooling="geometric"),
                  replace(top100, name="A14_unweighted_classes", parent=top100.name, balance=0.),
                  replace(top100, name="A15_sqrt_balance", parent=top100.name, balance=.5)])
    for i, model in enumerate(["logreg", "svm", "extratrees", "randomforest", "histgb", "lda", "gaussian_nb", "hier_catboost", "hier_logreg"], 16):
        specs.append(replace(top100, name=f"A{i:02d}_{model}", parent=top100.name, model=model))
    specs.append(replace(top100, name="A25_svm_pca25", parent="A17_svm", model="svm", pca=25))
    specs.append(replace(top100, name="A26_mutual_info", parent=top100.name, ranking="mutual_info"))
    return specs


def suggest_spec(trial):
    model = trial.suggest_categorical("model", ["catboost", "logreg", "svm", "extratrees", "randomforest", "histgb", "lda", "hier_catboost", "hier_logreg"])
    if model == "lda":
        representation = trial.suggest_categorical("lda_representation", ["subject_v1", "subject_plus"])
    else:
        representation = trial.suggest_categorical("representation", ["subject_v1", "subject_plus", "window14", "window28"])
    spec = Spec(name=f"T{trial.number:04d}", model=model, representation=representation,
                view=trial.suggest_categorical("view", ["all", "physiology", "no_quality", "activity", "sleep"]),
                balance=trial.suggest_categorical("balance", [0., .5, 1.]),
                top_k=trial.suggest_categorical("top_k", [20, 50, 100, 200, "all"]),
                correlation_limit=trial.suggest_categorical("correlation_limit", [.95, 1.]))
    if representation.startswith("window"):
        spec.pooling = trial.suggest_categorical("pooling", ["mean", "geometric"])
    family = model.removeprefix("hier_")
    if family == "catboost":
        spec.params = dict(iterations=trial.suggest_categorical("cb_iterations", [300, 600, 1000]),
                           depth=trial.suggest_int("cb_depth", 3, 6),
                           learning_rate=trial.suggest_float("cb_lr", .015, .1, log=True),
                           l2_leaf_reg=trial.suggest_float("cb_l2", 1., 80., log=True),
                           random_strength=trial.suggest_float("cb_random_strength", .05, 3., log=True),
                           boosting_type=trial.suggest_categorical("cb_boosting", ["Ordered", "Plain"]))
    elif family == "logreg":
        spec.params = dict(C=trial.suggest_float("lr_C", 1e-4, 10., log=True))
    elif family == "svm":
        spec.params = dict(C=trial.suggest_float("svm_C", .01, 100., log=True),
                           gamma=trial.suggest_categorical("svm_gamma", ["scale", .0001, .001, .01, .1]))
    elif family in {"extratrees", "randomforest"}:
        spec.params = dict(n_estimators=500,
                           min_samples_leaf=trial.suggest_int("forest_leaf", 1, 12),
                           max_depth=trial.suggest_categorical("forest_depth", [3, 5, 8, None]),
                           max_features=trial.suggest_categorical("forest_features", ["sqrt", .3, .7, 1.]))
    elif family == "histgb":
        spec.params = dict(max_iter=trial.suggest_categorical("hgb_iterations", [150, 300, 600]),
                           max_leaf_nodes=trial.suggest_categorical("hgb_leaves", [3, 7, 15]),
                           min_samples_leaf=trial.suggest_int("hgb_min_leaf", 5, 25),
                           learning_rate=trial.suggest_float("hgb_lr", .02, .15, log=True),
                           l2_regularization=trial.suggest_float("hgb_l2", .1, 50, log=True))
    elif family == "lda":
        spec.params = dict(shrinkage=trial.suggest_categorical("lda_shrinkage", ["auto", .1, .5, .9]))
    if family in {"logreg", "svm", "lda"}:
        spec.pca = trial.suggest_categorical("pca", [0, 10, 25])
    return spec


def evaluate_spec(data, labels, folds, spec, directory, config, seeds):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = spec.to_dict()
    settings.pop("name")
    settings.pop("parent")
    signature = {"spec": settings, "data": data.fingerprint, "source": source_signature(), "versions": versions(),
                 "labels": labels.to_dict(), "folds": [[list(a), list(b)] for a, b in folds], "seeds": list(seeds)}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = directory / f"{key}.npz"
    started = time.monotonic()
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            scores = saved["scores"]
            counts = saved["feature_counts"].tolist()
            fold_ids = saved["fold_ids"]
    else:
        scores = np.full((len(labels), 3), np.nan)
        fold_ids = np.full(len(labels), -1, int)
        counts = []
        rep = data.representations[spec.representation]
        for fold, (tr, te) in enumerate(folds):
            assert not set(tr).intersection(te)
            assert not set(rep.groups[rep.rows_for(tr)]).intersection(rep.groups[rep.rows_for(te)])
            predictions = []
            for seed in seeds:
                with threadpool_limits(limits=config.threads):
                    fitted = fit_model(rep, data.labels, tr, spec, seed=int(seed) + 101 * fold, threads=config.threads)
                    assert not set(fitted.training_subjects).intersection(te)
                    predictions.append(fitted.predict(rep, te))
                counts.append(len(fitted.processor.columns_))
            pos = labels.index.get_indexer(te)
            scores[pos] = np.mean(predictions, axis=0)
            fold_ids[pos] = fold
        assert np.isfinite(scores).all() and (fold_ids >= 0).all()
        temporary = cache.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, scores=scores, feature_counts=counts, fold_ids=fold_ids)
        temporary.replace(cache)
    scores = normalize_scores(scores)
    offsets, tuned = select_policy(labels.to_numpy(), scores, enabled=config.tune_decisions)
    raw = measure(labels.to_numpy(), scores)
    return {"spec": spec, "scores": scores, "offsets": offsets, "raw": raw, "tuned": tuned,
            "objective": objective(tuned), "mean_features": float(np.mean(counts)), "fold_ids": fold_ids,
            "seconds": time.monotonic() - started}


def results_table(results):
    rows = []
    for r in results:
        s = r["spec"]
        rows.append({"name": s.name, "parent": s.parent, "representation": s.representation,
                     "view": s.view, "model": s.model, "balance": s.balance, "top_k": s.top_k,
                     "objective": r["objective"], "mean_features": r["mean_features"], "seconds": r["seconds"],
                     **{f"raw_{k}": v for k, v in r["raw"].items()},
                     **{f"tuned_{k}": v for k, v in r["tuned"].items()}})
    table = pd.DataFrame(rows)
    by_name = table.set_index("name")
    for metric in ["roc_auc_macro_ovr", "macro_recall", "macro_f1"]:
        table[f"delta_raw_{metric}_vs_parent"] = [
            row[f"raw_{metric}"] - by_name.loc[row["parent"], f"raw_{metric}"]
            if row["parent"] in by_name.index else np.nan for row in rows]
    return table


def choose_ensemble(results, labels, config):
    diverse = {}
    for r in sorted(results, key=lambda x: x["objective"], reverse=True):
        s = r["spec"]
        diverse.setdefault((s.model, s.representation, s.view), r)
    pool = list(diverse.values())[:12]
    indices = [0]
    current = pool[0]["scores"]
    offsets, metrics = select_policy(labels.to_numpy(), current, enabled=config.tune_decisions)
    score = objective(metrics)
    history = [{"step": 1, "member": pool[0]["spec"].name, "objective": score}]
    for step in range(2, config.ensemble_steps + 1):
        winner = None
        for j, r in enumerate(pool):
            if j not in indices and len(set(indices)) >= config.max_ensemble_members:
                continue
            proposal = (current * len(indices) + r["scores"]) / (len(indices) + 1)
            bias, m = select_policy(labels.to_numpy(), proposal, enabled=config.tune_decisions)
            value = objective(m)
            if value > score + 1e-7 and (winner is None or value > winner[0]):
                winner = (value, j, proposal, bias, m)
        if winner is None:
            break
        score, j, current, offsets, metrics = winner
        indices.append(j)
        history.append({"step": step, "member": pool[j]["spec"].name, "objective": score})
    counts = Counter(indices)
    selected = [pool[i] for i in counts]
    weights = np.array([counts[i] / len(indices) for i in counts])
    return selected, weights, offsets, current, metrics, history


def run_search(data, labels, folds, directory, config, trial_budget):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fold_table(labels, folds).to_csv(directory / "fold_counts.csv", index=False)
    dump_json(directory / "fold_subjects.json", [{"train": list(a), "test": list(b)} for a, b in folds])
    results = []
    for i, spec in enumerate(ablation_specs(), 1):
        print(f"{directory.name}: ablation {i}/27 {spec.name}", flush=True)
        results.append(evaluate_spec(data, labels, folds, spec, directory / "cv_cache", config, (config.seed,)))
        results_table(results).to_csv(directory / "ablation.csv", index=False)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="subject_auc_recall",
                               storage=f"sqlite:///{(directory / 'optuna.sqlite3').resolve()}", load_if_exists=True,
                               sampler=optuna.samplers.TPESampler(seed=config.seed, n_startup_trials=15))
    # Recover completed trials after interruption. Their OOF matrices are cached.
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            spec = Spec(**trial.user_attrs["spec"])
            results.append(evaluate_spec(data, labels, folds, spec, directory / "cv_cache", config, (config.seed,)))

    def trial_objective(trial):
        spec = suggest_spec(trial)
        trial.set_user_attr("spec", spec.to_dict())
        result = evaluate_spec(data, labels, folds, spec, directory / "cv_cache", config, (config.seed,))
        results.append(result)
        trial.set_user_attr("metrics", result["tuned"])
        results_table(results).to_csv(directory / "search_results.csv", index=False)
        print(f"{directory.name}: trial {trial.number + 1}/{trial_budget}, "
              f"OOF AUC={result['tuned']['roc_auc_macro_ovr']:.3f}, "
              f"tuned recall={result['tuned']['macro_recall']:.3f}", flush=True)
        return result["objective"]

    remaining = max(0, trial_budget - len(study.trials))
    if remaining:
        study.optimize(trial_objective, n_trials=remaining, n_jobs=1,
                       catch=(ValueError, np.linalg.LinAlgError))
    study.trials_dataframe().to_csv(directory / "optuna_trials.csv", index=False)
    results_table(results).to_csv(directory / "search_results.csv", index=False)
    selected, weights, offsets, scores, metrics, history = choose_ensemble(results, labels, config)
    if tuple(config.final_seeds) != (config.seed,):
        refined = []
        for r in selected:
            print(f"{directory.name}: multi-seed OOF refinement: {r['spec'].name}", flush=True)
            refined.append(evaluate_spec(data, labels, folds, r["spec"], directory / "cv_cache", config, config.final_seeds))
        scores = normalize_scores(sum(w * r["scores"] for w, r in zip(weights, refined)))
        offsets, metrics = select_policy(labels.to_numpy(), scores, enabled=config.tune_decisions)
        selected = refined
    selection = {"specs": [r["spec"] for r in selected], "weights": weights,
                 "offsets": offsets, "scores": scores, "metrics": metrics,
                 "history": history, "seeds": list(config.final_seeds)}
    dump_json(directory / "selected.json", {"specs": [s.to_dict() for s in selection["specs"]],
                                           "weights": weights.tolist(), "offsets": offsets.tolist(),
                                           "metrics_development_only": metrics, "ensemble_history": history,
                                           "seeds": list(config.final_seeds),
                                           "note": "Selection and threshold tuning reused these OOF labels; these are not unbiased performance estimates."})
    pred = (np.log(scores) + offsets).argmax(axis=1)
    save_predictions(directory / "selected_development_oof.csv", labels, scores, pred)
    return selection


def fit_selection(data, train_subjects, selection, config):
    models, weights = [], []
    for spec, weight in zip(selection["specs"], selection["weights"]):
        for seed in selection["seeds"]:
            with threadpool_limits(limits=config.threads):
                fitted = fit_model(data.representations[spec.representation], data.labels,
                                   train_subjects, spec, seed=int(seed), threads=config.threads)
            models.append(fitted)
            weights.append(weight / len(selection["seeds"]))
    return EnsembleModel(models, np.array(weights), selection["offsets"], dict(fe.CFG),
                         {"source_sha256": source_signature(), "versions": versions(),
                          "training_subject_count": len(train_subjects), "MMSE_used": False})


def save_predictions(path, labels, scores, prediction, fold_ids=None):
    frame = pd.DataFrame(scores, index=labels.index, columns=[f"score_{n}" for n in NAMES])
    frame.index.name = "subject_hash"
    frame["true"] = np.array(NAMES)[labels.to_numpy()]
    frame["predicted"] = np.array(NAMES)[prediction]
    if fold_ids is not None:
        frame["fold"] = np.asarray(fold_ids) + 1
    frame.to_csv(path)


def nested_evaluation(data, directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    folds = subject_folds(data.labels, config.outer_folds, config.seed + 7301)
    fold_table(data.labels, folds).to_csv(directory / "outer_fold_counts.csv", index=False)
    scores = np.full((len(data.labels), 3), np.nan)
    pred = np.full(len(data.labels), -1, int)
    fold_ids = np.full(len(data.labels), -1, int)
    metrics = []
    for i, (tr, te) in enumerate(folds):
        print(f"Nested outer fold {i + 1}/{len(folds)}: {len(tr)} training / {len(te)} test subjects", flush=True)
        inner_labels = data.labels.loc[tr]
        inner = subject_folds(inner_labels, config.inner_folds, config.seed + 1000 + i)
        selection = run_search(data, inner_labels, inner, directory / f"outer_{i+1}", config, config.nested_trials)
        fitted = fit_selection(data, tr, selection, config)
        assert all(not set(model.training_subjects).intersection(te) for model in fitted.models)
        p, h = fitted.predict(data.representations, te)
        pos = data.labels.index.get_indexer(te)
        scores[pos], pred[pos], fold_ids[pos] = p, h, i
        m = measure(data.labels.loc[te].to_numpy(), p, h)
        metrics.append(dict(fold=i + 1, **m))
        pd.DataFrame(metrics).to_csv(directory / "outer_metrics.csv", index=False)
        save_predictions(directory / f"outer_{i+1}" / "outer_predictions.csv", data.labels.loc[te], p, h)
    assert np.isfinite(scores).all() and (pred >= 0).all()
    summary = measure(data.y, scores, pred)
    dump_json(directory / "metrics.json", summary)
    save_predictions(directory / "outer_oof_predictions.csv", data.labels, scores, pred, fold_ids)
    intervals(data.y, scores, pred, repeats=config.bootstrap_repeats, seed=config.seed).to_csv(directory / "bootstrap_ci.csv", index=False)
    return summary, scores, pred


def run_experiment(config=None, data_root=None):
    config = Config() if config is None else config
    if not config.final_seeds or config.threads < 1:
        raise ValueError("At least one model seed and a positive thread count are required.")
    _, paths = fe.resolve_paths(data_root)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    training = load_dataset(paths["train"], config.cache_dir)
    manifest = {"config": asdict(config), "training_data": training.fingerprint, "source": source_signature(),
                "versions": versions(), "python": sys.version, "platform": platform.platform(),
                "class_names": NAMES, "selection_metric": "min(macro_OVR_AUC, macro_recall) with tiny tie-breaks",
                "validation_policy": "Previously reported official validation. Never used to select v2 parameters, weights or thresholds."}
    # JSON roundtrip normalizes tuples before comparison when resuming.
    manifest = json.loads(json.dumps(manifest))
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Run directory has a different code/data/config manifest. Choose a new Config.output_dir; do not mix experiments.")
    dump_json(manifest_path, manifest)
    if (output / "summary.json").exists():
        print("This exact run is already complete; returning saved results.", flush=True)
        return json.loads((output / "summary.json").read_text())
    dump_json(output / "training_data_audit.json", training.audit)
    nested = None
    if config.run_nested:
        nested, nested_scores, nested_pred = nested_evaluation(training, output / "nested", config)
    development = subject_folds(training.labels, config.development_folds, config.seed)
    selection = run_search(training, training.labels, development, output / "development", config, config.n_trials)
    # Lock every design decision BEFORE loading/evaluating official validation.
    final_model = fit_selection(training, training.subjects, selection, config)
    joblib.dump(final_model, output / "candidate_model.joblib", compress=3)
    validation = load_dataset(paths["val"], config.cache_dir)
    if set(training.subjects).intersection(validation.subjects):
        raise ValueError("Training/Validation subject overlap: refusing evaluation.")
    dump_json(output / "validation_data_audit.json", {"audit": validation.audit, "fingerprint": validation.fingerprint})
    scores, prediction = final_model.predict(validation.representations, validation.subjects)
    holdout = measure(validation.y, scores, prediction)
    save_predictions(output / "validation_predictions.csv", validation.labels, scores, prediction)
    intervals(validation.y, scores, prediction, repeats=config.bootstrap_repeats, seed=config.seed).to_csv(output / "validation_bootstrap_ci.csv", index=False)
    validation_status = target_status(holdout, config.threshold)
    nested_status = target_status(nested, config.threshold) if nested is not None else None
    confirmed = bool(validation_status["macro_target_met"] and nested_status is not None and nested_status["macro_target_met"])
    summary = {"development_oof_selection_biased": selection["metrics"], "nested_outer_oof": nested,
               "official_validation_reused_from_v1": holdout, "validation_target": validation_status,
               "nested_target": nested_status, "both_evaluations_macro_target_met": confirmed,
               "threshold": config.threshold, "selected_specs": [s.to_dict() for s in selection["specs"]],
               "weights": selection["weights"].tolist(), "offsets": selection["offsets"].tolist(),
               "saved_candidate": str(output / "candidate_model.joblib"),
               "caution": "Small MCI/DEM counts. Validation was viewed in v1; it is not a new untouched external cohort. Nested outer CV uses a smaller search budget and training size than the final fit."}
    if validation_status["macro_target_met"]:
        joblib.dump(final_model, output / "validation_target_met_model.joblib", compress=3)
    if confirmed:
        joblib.dump(final_model, output / "cv_and_validation_target_met_model.joblib", compress=3)
    from .reporting import write_report
    write_report(output, summary, validation.y, scores, prediction,
                 None if nested is None else (training.y, nested_scores, nested_pred))
    # Written last: an interruption before report completion can be resumed.
    dump_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Extract/cache features and audit only. Never fit a classifier.")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", default="results_v2/run_full")
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--nested-trials", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-nested", action="store_true", help="Development only; cannot substantiate a nested-CV target.")
    args = parser.parse_args()
    cfg = Config(output_dir=args.output_dir, n_trials=args.trials, nested_trials=args.nested_trials,
                 threads=args.threads, run_nested=not args.skip_nested)
    if args.prepare_only:
        _, paths = fe.resolve_paths(args.data_root)
        for split in ["train", "val"]:
            data = load_dataset(paths[split], cfg.cache_dir)
            print(split, data.labels.value_counts().to_dict(), {k: r.X.shape for k, r in data.representations.items()})
    else:
        run_experiment(cfg, args.data_root)


if __name__ == "__main__":
    main()
