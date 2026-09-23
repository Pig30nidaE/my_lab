"""Label-free personal features, with literature-inspired temporal variability.

The HR-drop definitions are explicit reconstructions, not claims of exact paper
reproduction. No subject is removed based on diagnosis, outliers or model errors.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, trim_mean

from lifelog_v2 import feature_engineering as fe
from lifelog_v2.data import Dataset, Representation, load_dataset


def safe_ratio(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and abs(b) > 1e-8 else np.nan


def hr_features(row):
    hr = fe.parse_sequence(row.get(fe.SEQ_COLS['sleep']['hr']), positive_only=True)
    valid = hr[np.isfinite(hr)]
    first = hr[:3]
    first = first[np.isfinite(first)]
    out = {}
    # Preserve temporal positions: missing early epochs are not shifted forward.
    baseline = first.mean() if len(first) >= 2 else np.nan
    low = valid.min() if len(valid) >= 4 else np.nan
    low05 = np.quantile(valid, .05) if len(valid) >= 4 else np.nan
    nrem_hours = (fe.number(row.get('sleep_deep')) + fe.number(row.get('sleep_light'))) / 3600
    total_hours = fe.number(row.get('sleep_total')) / 3600
    for name, denominator in [('ratio', baseline), ('per_nrem_hour', nrem_hours), ('per_sleep_hour', total_hours)]:
        out[f'hr_drop_{name}'] = safe_ratio(baseline - low, denominator)
        out[f'hr_drop05_{name}'] = safe_ratio(baseline - low05, denominator)
    # Separate alternative matching the team's first-three-valid reconstruction.
    legacy = valid[:3].mean() if len(valid) >= 4 else np.nan
    out['hr_drop_firstvalid_ratio'] = safe_ratio(legacy - low, legacy)
    out['hr_baseline_first15min'] = baseline
    out['hr_robust_range'] = np.quantile(valid, .95) - low05 if len(valid) >= 4 else np.nan
    if len(hr) >= 12 and np.isfinite(hr).sum() >= 12:
        good = np.isfinite(hr)
        time = np.arange(len(hr), dtype=float) * 5 / 60
        out['hr_slope_per_hour'] = np.polyfit(time[good], hr[good], 1)[0]
        # Location of the minimum in the original night, not compressed valid data.
        out['hr_min_relative_time'] = float(np.nanargmin(hr) / max(len(hr) - 1, 1))
    else:
        out['hr_slope_per_hour'] = out['hr_min_relative_time'] = np.nan
    start = fe.local_timestamp(row.get('sleep_bedtime_start'))
    clock = (fe.clock_hours(start) + float(np.nanargmin(hr))*5/60) % 24 if pd.notna(start) and len(valid)>=4 else np.nan
    out['hr_min_clock_sin'] = np.sin(clock*np.pi/12)
    out['hr_min_clock_cos'] = np.cos(clock*np.pi/12)
    return {f'sleep__paper_{k}': v for k, v in out.items()}


def load_daily(path, modality):
    """Explicit wearable allowlist; never read any CognitiveFunction/MMSE file."""
    path = Path(path)
    if 'mmse' in str(path).lower() or 'cognitivefunction' in str(path).lower():
        raise ValueError('MMSE/cognitive input files are forbidden.')
    allowed = ['EMAIL'] + (fe.ACT_NUM if modality == 'activity' else fe.SLEEP_NUM)
    allowed += fe.TIME_COLS[modality] + list(fe.SEQ_COLS[modality].values())
    records = []
    for chunk in pd.read_csv(path, usecols=allowed, dtype=str, encoding='utf-8-sig', chunksize=256):
        for row in chunk.to_dict('records'):
            result = fe.extract_day(row, modality)
            if modality == 'sleep':
                result.update(hr_features(row))
                start, end = [fe.local_timestamp(row[c]) for c in fe.TIME_COLS['sleep']]
                bed, wake = fe.clock_hours(start), fe.clock_hours(end)
                result['sleep__paper_bedtime_noon_unwrapped'] = bed + 24 if bed < 12 else bed
                result['sleep__paper_wake_hour'] = wake
                result['sleep__paper_interval_hours'] = (end-start).total_seconds()/3600 if pd.notna(start) and pd.notna(end) else np.nan
                midpoint = start + (end-start)/2 if pd.notna(start) and pd.notna(end) else pd.NaT
                mid = fe.clock_hours(midpoint)
                result['sleep__paper_midpoint_sin'] = np.sin(mid * np.pi / 12)
                result['sleep__paper_midpoint_cos'] = np.cos(mid * np.pi / 12)
                # Epoch count is quality, distinct from physiological variability.
                result['sleep__paper_nrem_fraction'] = safe_ratio(
                    fe.number(row.get('sleep_deep')) + fe.number(row.get('sleep_light')),
                    fe.number(row.get('sleep_total')))
            records.append(result)
    frame = pd.DataFrame(records).dropna(subset=['_date'])
    counts = frame.groupby(['subject', '_date']).size()
    if modality == 'sleep':
        frame = frame.sort_values(['subject', '_date', 'sleep__duration', '_start'], ascending=[True, True, False, True], kind='stable')
    else:
        frame = frame.sort_values(['subject', '_date', '_start'], kind='stable')
    frame = frame.drop_duplicates(['subject', '_date']).copy()
    if modality == 'sleep':
        frame['sleep__paper_episode_count'] = [float(counts.loc[(s, d)]) for s, d in zip(frame.subject, frame._date)]
    return frame.sort_values(['subject', '_date']).drop(columns='_start').reset_index(drop=True)


def temporal_summary(series):
    """Within-person calendar-day summaries, never crossing missing-day gaps."""
    series = series.astype(float).replace([np.inf, -np.inf], np.nan)
    finite = series.dropna().to_numpy()
    keys = ['mean', 'std', 'median', 'trimmed_mean', 'mode', 'minimum', 'maximum', 'range', 'mad',
            'cv', 'skew', 'kurtosis', 'q05', 'q95', 'rmssd', 'moving_range', 'lag1',
            'trend', 'bin_variance4', 'bin_change4']
    out = dict.fromkeys(keys, np.nan)
    for w in (7, 14, 28):
        out.update(dict.fromkeys([f'stv{w}', f'rolling_cv{w}', f'rolling_range{w}'], np.nan))
    if not len(finite):
        return out
    mean = finite.mean()
    std = finite.std(ddof=1) if len(finite) > 1 else 0.
    median = np.median(finite)
    rounded, counts = np.unique(np.round(finite,3),return_counts=True)
    out.update(mean=mean, std=std, median=median, trimmed_mean=trim_mean(finite, .1),
               mode=rounded[np.argmax(counts)], minimum=finite.min(), maximum=finite.max(), range=np.ptp(finite),
               mad=np.median(np.abs(finite-median)), cv=safe_ratio(std, abs(mean)),
               q05=np.quantile(finite,.05), q95=np.quantile(finite,.95))
    if std > 1e-8 and len(finite) >= 8:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            out['skew'] = skew(finite, bias=False)
            out['kurtosis'] = kurtosis(finite, fisher=True, bias=False)
    calendar = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq='D'))
    delta = calendar.diff().dropna().to_numpy()
    if len(delta):
        out['rmssd'] = np.sqrt(np.mean(delta**2))
        out['moving_range'] = np.mean(np.abs(delta))
    a, b = calendar.to_numpy()[:-1], calendar.to_numpy()[1:]
    out['lag1'] = fe.safe_corr(a, b)
    elapsed = np.arange(len(calendar), dtype=float)
    good = np.isfinite(calendar.to_numpy())
    if good.sum() >= 4 and np.ptp(elapsed[good]) > 0:
        out['trend'] = np.polyfit(elapsed[good], calendar.to_numpy()[good], 1)[0]
    bin_id = np.minimum((elapsed * 4 / max(len(calendar), 1)).astype(int), 3)
    bins = pd.Series(calendar.to_numpy()).groupby(bin_id).mean().reindex(range(4))
    if bins.notna().all():
        out['bin_variance4'] = bins.var(ddof=1)
        out['bin_change4'] = np.abs(np.diff(bins)).mean()
    for w in (7, 14, 28):
        rolling = calendar.rolling(w, min_periods=max(4, w//2))
        sd, mu = rolling.std(ddof=1), rolling.mean()
        out[f'stv{w}'] = sd.mean()
        out[f'rolling_cv{w}'] = (sd / mu.abs().where(mu.abs() > 1e-8)).mean()
        out[f'rolling_range{w}'] = (rolling.max()-rolling.min()).mean()
    return out


def build_extra(activity, sleep, subjects):
    rows, meta = [], {}
    for subject in subjects:
        row = {}
        for modality, frame in [('activity', activity), ('sleep', sleep)]:
            own = frame.loc[frame.subject.eq(subject)].set_index('_date')
            # Acquisition metadata is permitted by the two user constraints.
            # Its own family enables explicit with/without-context comparison.
            # No diagnosis date or identifier-derived number is read or encoded.
            if len(own):
                epoch = (own.index-pd.Timestamp('2000-01-01')).days.to_numpy(float)
                phase = 2*np.pi*(own.index.dayofyear.to_numpy()-1)/365.25
                context = {'first_epochday':epoch.min(),'last_epochday':epoch.max(),
                           'median_epochday':np.median(epoch),
                           'season_sin':np.mean(np.sin(phase)),'season_cos':np.mean(np.cos(phase)),
                           'weekend_fraction':float(np.mean(own.index.dayofweek>=5))}
            else:
                context = dict.fromkeys(['first_epochday','last_epochday','median_epochday','season_sin','season_cos','weekend_fraction'],np.nan)
            for key,value in context.items():
                name=f'context__{modality}__{key}'
                row[name]=value
                meta[name]={'modality':modality,'family':'collection_context','kind':'collection_context',
                            'description':f'Acquisition timing, not a physiological biomarker: {key}'}
            for col in own.columns:
                if not col.startswith(modality + '__'):
                    continue
                # Daily quality fields are already covered in v2; limit duplicates.
                is_quality = any(token in col for token in ['valid_fraction', '_length', 'interval_minutes', 'episode_count', 'non_wear'])
                if is_quality and 'episode_count' not in col:
                    continue
                for stat, value in temporal_summary(own[col]).items():
                    short = col.removeprefix(modality + '__')
                    name = f'v3__{modality}__{short}__{stat}'
                    row[name] = value
                    family = 'quality' if is_quality else ('vendor' if 'score' in col else 'physiology')
                    meta[name] = {'modality': modality, 'family': family, 'kind': 'temporal_v3',
                                  'description': f'{col}: within-person {stat}; calendar windows where applicable'}
        rows.append(row)
    X = pd.DataFrame(rows, index=subjects).replace([np.inf, -np.inf], np.nan).astype('float32')
    return X, pd.DataFrame.from_dict(meta, orient='index').reindex(X.columns)


def load_split(paths, cache_dir='results_v3/cache', legacy_cache='results_v2/cache'):
    base = load_dataset(paths, legacy_cache)
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    signature = dict(base.fingerprint, v3_feature_source=source, numpy=np.__version__, pandas=pd.__version__)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f'features_{key}.joblib'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return joblib.load(path)
    print('Extracting V3 heart-rate and calendar variability features...', flush=True)
    a = load_daily(paths['activity'], 'activity')
    s = load_daily(paths['sleep'], 'sleep')
    extra, meta = build_extra(a, s, base.subjects)
    legacy = base.representations['subject_plus']
    legacy_meta = legacy.meta.copy()
    legacy_meta['kind'] = 'v2'
    X = pd.concat([legacy.X, extra], axis=1)
    metadata = pd.concat([legacy_meta, meta]).reindex(X.columns)
    assert X.columns.is_unique and X.index.is_unique and metadata.family.notna().all()
    forbidden = ['mmse', 'diag', 'email', 'subject', 'timestamp', '_date']
    assert not any(any(x in c.lower() for x in forbidden) for c in X.columns)
    representations = dict(base.representations)
    representations['subject'] = Representation(X, base.subjects, metadata)
    result = Dataset(base.labels, representations, base.audit, signature)
    joblib.dump(result, path, compress=3)
    return result


def inference_data(activity_csv, sleep_csv):
    from lifelog_v2.data import inference_representations
    legacy, subjects = inference_representations(activity_csv, sleep_csv, windows=(14,28))
    a, s = load_daily(activity_csv, 'activity'), load_daily(sleep_csv, 'sleep')
    extra, meta = build_extra(a, s, subjects)
    old = legacy['subject_plus']
    old_meta = old.meta.copy()
    old_meta['kind'] = 'v2'
    legacy['subject'] = Representation(pd.concat([old.X, extra], axis=1), subjects,
                                       pd.concat([old_meta, meta]))
    return legacy, subjects


def prepare(data_root=None):
    _, paths = fe.resolve_paths(data_root)
    train, val = load_split(paths['train']), load_split(paths['val'])
    assert not set(train.subjects).intersection(val.subjects)
    summary = {'classifier_training_executed': False, 'subject_overlap': 0, 'splits': {}}
    for name, ds in [('Training', train), ('SelectionValidation', val)]:
        summary['splits'][name] = {'subjects': len(ds.subjects),
            'class_counts': {fe.CLASS_NAMES[int(k)]: int(v) for k,v in ds.labels.value_counts().items()},
            'representations': {k: {'rows':len(r.X), 'features':r.X.shape[1]} for k,r in ds.representations.items()}}
    Path('results_v3').mkdir(exist_ok=True)
    Path('results_v3/preparation.json').write_text(json.dumps(summary, indent=2))
    train.representations['subject'].meta.to_csv('results_v3/feature_dictionary.csv', index_label='feature')
    print(json.dumps(summary, indent=2), flush=True)
    return train, val


if __name__ == '__main__':
    prepare()
