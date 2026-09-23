"""Personal feature construction. No population fitting and no MMSE access."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from . import feature_engineering as fe


@dataclass
class Representation:
    X: pd.DataFrame
    groups: np.ndarray
    meta: pd.DataFrame

    def rows_for(self, subjects):
        return np.flatnonzero(np.isin(self.groups, np.asarray(subjects)))


@dataclass
class Dataset:
    labels: pd.Series
    representations: dict[str, Representation]
    audit: list[dict]
    fingerprint: dict

    @property
    def subjects(self):
        return self.labels.index.to_numpy()

    @property
    def y(self):
        return self.labels.to_numpy(dtype=int)


def fingerprint(paths):
    """Hash only explicitly allowlisted activity/sleep files and their labels."""
    result = {}
    for name, path in sorted(paths.items()):
        if name not in {"activity", "sleep", "activity_label", "sleep_label"}:
            raise ValueError(f"Non-allowlisted input: {name}")
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[name] = {"name": Path(path).name, "sha256": digest.hexdigest()}
    for source in [Path(__file__), Path(fe.__file__)]:
        result[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    result["extraction_config"] = dict(fe.CFG)
    return result


def enhanced_summary(activity, sleep, base):
    """Add within-person distribution, weekday and hourly rhythm summaries."""
    rows, metadata = [], {}
    for subject in base.index:
        out = {}
        for modality, frame in [("activity", activity), ("sleep", sleep)]:
            g = frame.loc[frame.subject.eq(subject)].sort_values("_date")
            for column, m in fe.DAILY_META.items():
                if m["modality"] != modality or column not in g:
                    continue
                values = g[column].to_numpy(float)
                finite = values[np.isfinite(values)]
                median = np.median(finite) if len(finite) else np.nan
                mean = np.mean(finite) if len(finite) else np.nan
                std = np.std(finite) if len(finite) else np.nan
                stats = {
                    "mean": mean, "std": std,
                    "mad": np.median(np.abs(finite - median)) if len(finite) else np.nan,
                    "cv": std / abs(mean) if np.isfinite(mean) and abs(mean) > 1e-6 else np.nan,
                }
                if len(g):
                    weekend = g._date.dt.dayofweek.to_numpy() >= 5
                    w = values[weekend & np.isfinite(values)]
                    d = values[~weekend & np.isfinite(values)]
                    stats["weekend_delta"] = np.mean(w) - np.mean(d) if len(w) >= 2 and len(d) >= 2 else np.nan
                else:
                    stats["weekend_delta"] = np.nan
                for stat, value in stats.items():
                    name = f"{column}__extra_{stat}"
                    out[name] = value
                    metadata[name] = dict(m, description=f"{column}: within-person {stat}")
            if modality == "activity":
                for h in range(24):
                    a = g[f"_hour_{h:02d}"].dropna().to_numpy(float)
                    for stat, value in [("mean", a.mean() if len(a) else np.nan),
                                        ("std", a.std() if len(a) else np.nan)]:
                        name = f"activity__hour_{h:02d}__{stat}"
                        out[name] = value
                        metadata[name] = {"modality": modality, "family": "physiology",
                                          "description": f"Local hour {h}: across-day MET {stat}"}
        rows.append(out)
    extra = pd.DataFrame(rows, index=base.index)
    combined = pd.concat([base, extra], axis=1).replace([np.inf, -np.inf], np.nan).astype("float32")
    return combined, metadata


def one_representation(activity, sleep, enhanced=False):
    X, a, s, days = fe.build_subjects(activity, sleep)
    meta = pd.DataFrame.from_dict(fe.FEATURE_META_RECORDS, orient="index").loc[X.columns]
    if enhanced:
        X, extra_meta = enhanced_summary(a, s, X)
        meta = pd.concat([meta, pd.DataFrame.from_dict(extra_meta, orient="index")]).loc[X.columns]
    assert X.index.is_unique and X.columns.is_unique
    # Identity and absolute dates must never appear as predictors.
    forbidden = ("mmse", "diag", "email", "subject", "sample_id", "_date", "timestamp")
    assert not any(any(token in c.lower() for token in forbidden) for c in X.columns)
    return Representation(X, X.index.to_numpy(), meta)


def window_representation(activity, sleep, template, width):
    """Nonoverlapping calendar windows; every window stays with its subject.

    Windows with <7 observed days are excluded. A person with no eligible window
    falls back to their full-period summary, preserving the evaluation cohort.
    """
    matrices, group_arrays, fallback = [], [], 0
    dates = pd.concat([activity[["subject", "_date"]], sleep[["subject", "_date"]]]).drop_duplicates()
    for subject in template.groups:
        a = activity.loc[activity.subject.eq(subject)]
        s = sleep.loc[sleep.subject.eq(subject)]
        own_dates = dates.loc[dates.subject.eq(subject), "_date"]
        start = own_dates.min()
        blocks = ((own_dates - start).dt.days // width).unique()
        found = False
        for block in sorted(blocks):
            lo = start + pd.Timedelta(days=int(block) * width)
            hi = lo + pd.Timedelta(days=width)
            mask = own_dates.ge(lo) & own_dates.lt(hi)
            if int(mask.sum()) < fe.CFG["min_days"]:
                continue
            aa = a.loc[a._date.ge(lo) & a._date.lt(hi)]
            ss = s.loc[s._date.ge(lo) & s._date.lt(hi)]
            rep = one_representation(aa, ss, enhanced=True)
            matrices.append(rep.X.reindex(columns=template.X.columns))
            group_arrays.append(np.array([subject]))
            found = True
        if not found:
            matrices.append(template.X.loc[[subject]])
            group_arrays.append(np.array([subject]))
            fallback += 1
    X = pd.concat(matrices, ignore_index=True)
    groups = np.concatenate(group_arrays)
    assert set(groups) == set(template.groups)
    audit = {"representation": f"window{width}", "rows": len(X),
             "subjects": len(set(groups)), "full_period_fallback_subjects": fallback}
    return Representation(X, groups, template.meta.copy()), audit


def load_dataset(paths, cache_dir, *, windows=(14, 28)):
    """Run explicitly by the user. Cached features are local deterministic data."""
    signature = fingerprint(paths)
    signature["windows"] = list(windows)
    cache_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"features_{cache_key}.joblib"
    if target.exists():
        print(f"Using feature cache: {target.name}", flush=True)
        return joblib.load(target)
    fe.AUDIT.clear()
    fe.DAILY_META.clear()
    fe.FEATURE_META_RECORDS.clear()
    labels = fe.read_labels(paths).map(fe.LABEL_MAP).astype(int)
    activity = fe.load_daily(paths["activity"], "activity", "requested_split")
    sleep = fe.load_daily(paths["sleep"], "sleep", "requested_split")
    activity = activity.loc[activity.subject.isin(labels.index)]
    sleep = sleep.loc[sleep.subject.isin(labels.index)]
    v1 = one_representation(activity, sleep)
    plus = one_representation(activity, sleep, enhanced=True)
    labels = labels.reindex(v1.groups)
    assert labels.notna().all() and labels.index.is_unique
    audit = list(fe.AUDIT)
    reps = {"subject_v1": v1, "subject_plus": plus}
    for width in windows:
        print(f"Constructing {width}-day within-person windows...", flush=True)
        reps[f"window{width}"], record = window_representation(activity, sleep, plus, width)
        audit.append(record)
    for rep in reps.values():
        assert set(rep.groups) == set(labels.index)
        assert rep.X.columns.equals(rep.meta.index)
    result = Dataset(labels, reps, audit, signature)
    joblib.dump(result, target, compress=3)
    return result


def inference_representations(activity_csv, sleep_csv, *, windows=(14, 28)):
    """Feature-only path: no diagnosis labels are required for new subjects."""
    fe.DAILY_META.clear()
    fe.FEATURE_META_RECORDS.clear()
    a = fe.load_daily(Path(activity_csv), "activity", "inference")
    s = fe.load_daily(Path(sleep_csv), "sleep", "inference")
    v1 = one_representation(a, s)
    plus = one_representation(a, s, enhanced=True)
    reps = {"subject_v1": v1, "subject_plus": plus}
    for width in windows:
        reps[f"window{width}"], _ = window_representation(a, s, plus, width)
    return reps, v1.groups
