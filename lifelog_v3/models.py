"""Estimator fitting sees Training subjects only; selection is external."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import importlib.util
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer, StandardScaler, RobustScaler
from sklearn.svm import SVC
from catboost import CatBoostClassifier

from lifelog_v2.modeling import subject_weights


@dataclass
class Spec:
    task: str = 'flat'
    family: str = 'catboost'
    representation: str = 'subject'
    view: str = 'all'
    top_k: int | str = 40
    selection: str = 'anova'
    transform: str = 'robust'
    correlation: float = 1.
    pca: int = 0
    balance: float = 1.
    augmentation: str = 'none'
    pooling: str = 'mean'
    seed: int = 2026
    params: dict = field(default_factory=dict)

    def dictionary(self):
        return asdict(self)


def families():
    names = ['catboost','logreg','svm','lda','extratrees','randomforest','histgb','rbf_ridge']
    for name in ['xgboost','lightgbm']:
        if importlib.util.find_spec(name):
            names.append(name)
    return names


def task_labels(labels, task):
    if task == 'flat':
        return labels.copy()
    if task == 'dem':
        return labels.eq(2).astype(int)
    if task == 'cn_mci':
        return labels.loc[labels.ne(2)].copy()
    raise ValueError(task)


def view_columns(meta, view):
    names = meta.index.to_series()
    variable = names.str.contains(r'__(?:std|cv|mad|iqr|rmssd|moving_range|stv\d+|rolling_cv\d+|rolling_range\d+|range|bin_variance4)(?:$|__)')
    if view == 'all':
        mask = np.ones(len(meta), bool)
    elif view == 'legacy':
        mask = ~names.str.startswith('v3__') & meta.family.ne('collection_context')
    elif view == 'temporal':
        mask = names.str.startswith('v3__')
    elif view == 'variability':
        mask = variable
    elif view == 'activity_variability':
        mask = meta.modality.eq('activity') & variable
    elif view in ['activity','sleep']:
        mask = meta.modality.eq(view)
    elif view == 'physiology':
        mask = meta.family.eq('physiology')
    elif view == 'no_context':
        mask = meta.family.ne('collection_context')
    elif view == 'context_only':
        mask = meta.family.eq('collection_context')
    elif view == 'paper_sleep':
        mask = names.str.startswith('v3__sleep__')
    elif view == 'hr_drop':
        mask = names.str.contains('paper_hr_')
    elif view == 'circadian':
        mask = names.str.contains('cosinor|regularity|midpoint|bedtime|wake_|hourly|high10|low5')
    elif view == 'dem_single':
        preferred = 'v3__activity__low__std'
        return [preferred] if preferred in meta.index else ['activity__low__extra_std']
    elif view == 'compact':
        desired = [
            'v3__activity__low__std','v3__activity__steps__std','v3__activity__inactive__std',
            'v3__activity__rest__mean','v3__sleep__paper_hr_drop_ratio__std',
            'v3__sleep__paper_hr_drop_ratio__rolling_cv7','v3__sleep__paper_hr_drop05_ratio__mad',
            'v3__sleep__paper_bedtime_noon_unwrapped__std','v3__sleep__paper_wake_hour__std',
            'v3__sleep__efficiency__mad','v3__sleep__total__stv7','v3__sleep__rmssd__cv',
            'activity__hourly_profile_stability','sleep__bedtime_regularity','sleep__wake_regularity']
        return [c for c in desired if c in meta.index]
    else:
        raise ValueError(view)
    return meta.index[np.asarray(mask)].tolist()


class Processor:
    def __init__(self, spec):
        self.spec = spec

    def fit(self, X, y, groups, meta):
        columns = view_columns(meta, self.spec.view)
        centers = X[columns].replace([np.inf,-np.inf],np.nan).groupby(np.asarray(groups)).mean()
        targets = pd.Series(y,index=groups).groupby(level=0).first().reindex(centers.index)
        usable = centers.notna().sum().ge(max(3,int(.1*len(centers)))) & centers.nunique().gt(1)
        centers = centers.loc[:,usable]
        if not centers.shape[1]:
            raise ValueError('No varying train-only features in this view.')
        medians = centers.median().fillna(0)
        filled = centers.fillna(medians)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            if self.spec.selection == 'mutual_info':
                ranks = mutual_info_classif(filled, targets, random_state=self.spec.seed)
            else:
                ranks = f_classif(filled,targets)[0]
        ranks = np.nan_to_num(ranks,nan=-np.inf,posinf=1e30)
        order = filled.columns[np.argsort(-ranks,kind='stable')].tolist()
        wanted = len(order) if self.spec.top_k == 'all' else min(int(self.spec.top_k),len(order))
        if self.spec.selection == 'rfe':
            # Supervised RFE uses training labels only, never selection labels.
            shortlist = order[:min(len(order),max(150,wanted*3))]
            scaled = StandardScaler().fit_transform(filled[shortlist])
            selector = RFE(LogisticRegression(C=.1,max_iter=2500),
                           n_features_to_select=min(wanted,len(shortlist)),step=.2)
            selector.fit(scaled, targets)
            selected = list(np.array(shortlist)[selector.support_])
        elif self.spec.correlation < 1:
            # Limit pair matrix to ranked candidates to avoid a huge p x p matrix.
            pool = order[:max(500,wanted*3)]
            corr = filled[pool].corr().abs().fillna(0)
            selected = []
            for c in pool:
                if not selected or corr.loc[c,selected].max() < self.spec.correlation:
                    selected.append(c)
                if len(selected) >= wanted:
                    break
        else:
            selected = order[:wanted]
        self.columns = selected
        self.medians = medians[selected]
        values = filled[selected].to_numpy(float)
        if self.spec.transform == 'signedlog':
            values = np.sign(values)*np.log1p(np.abs(values))
        self.scaler = (QuantileTransformer(n_quantiles=min(100,len(values)), output_distribution='normal',random_state=self.spec.seed)
                       if self.spec.transform == 'quantile' else
                       StandardScaler() if self.spec.transform == 'standard' else RobustScaler())
        values = np.clip(self.scaler.fit_transform(values),-30,30)
        self.pca_model = None
        if self.spec.pca:
            n = min(self.spec.pca, values.shape[1],len(values)-1)
            self.pca_model = PCA(n_components=n,svd_solver='full').fit(values)
        return self

    def transform(self, X):
        if not set(self.columns).issubset(X.columns):
            raise ValueError('Inference input is missing selected features.')
        values = X[self.columns].replace([np.inf,-np.inf],np.nan).fillna(self.medians).to_numpy(float)
        if self.spec.transform == 'signedlog':
            values = np.sign(values)*np.log1p(np.abs(values))
        values = np.clip(self.scaler.transform(values),-30,30)
        return self.pca_model.transform(values) if self.pca_model is not None else values


def estimator(spec, n_classes, threads=4):
    p, family = dict(spec.params), spec.family
    if family == 'catboost':
        defaults = dict(iterations=600,depth=4,learning_rate=.04,l2_leaf_reg=10,
                        boosting_type='Plain',border_count=64)
        defaults.update(p)
        return CatBoostClassifier(**defaults,loss_function='MultiClass' if n_classes==3 else 'Logloss',
                                 random_seed=spec.seed,thread_count=threads,verbose=False,allow_writing_files=False)
    if family == 'logreg':
        return LogisticRegression(C=p.get('C',.1),max_iter=4000,solver='lbfgs')
    if family == 'svm':
        return SVC(C=p.get('C',1),gamma=p.get('gamma','scale'),kernel='rbf',probability=False,
                   decision_function_shape='ovr',random_state=spec.seed,cache_size=512)
    if family == 'lda':
        return LinearDiscriminantAnalysis(solver='lsqr',shrinkage=p.get('shrinkage','auto'))
    if family in ['extratrees','randomforest']:
        cls = ExtraTreesClassifier if family=='extratrees' else RandomForestClassifier
        defaults = dict(n_estimators=500,min_samples_leaf=2,max_features='sqrt',max_depth=None)
        defaults.update(p)
        return cls(**defaults,n_jobs=threads,random_state=spec.seed)
    if family == 'histgb':
        defaults = dict(max_iter=350,max_leaf_nodes=7,min_samples_leaf=8,l2_regularization=5)
        defaults.update(p)
        return HistGradientBoostingClassifier(**defaults,early_stopping=False,random_state=spec.seed)
    if family == 'rbf_ridge':
        return KernelRidge(alpha=p.get('alpha',1),kernel='rbf',gamma=p.get('gamma',.01))
    if family == 'xgboost':
        from xgboost import XGBClassifier
        defaults = dict(n_estimators=600,max_depth=3,learning_rate=.035,min_child_weight=3,
                        reg_lambda=10,subsample=.85,colsample_bytree=.8)
        defaults.update(p)
        return XGBClassifier(**defaults,objective='multi:softprob' if n_classes==3 else 'binary:logistic',
                             tree_method='hist',n_jobs=threads,random_state=spec.seed)
    if family == 'lightgbm':
        from lightgbm import LGBMClassifier
        defaults = dict(n_estimators=600,num_leaves=7,learning_rate=.035,min_child_samples=5,
                        reg_lambda=10,verbosity=-1)
        defaults.update(p)
        return LGBMClassifier(**defaults,objective='multiclass' if n_classes==3 else 'binary',
                              n_jobs=threads,random_state=spec.seed)
    raise ValueError(family)


def augment_training(X, y, groups, method, seed):
    if method == 'none':
        return X,y,groups,{'method':'none','synthetic_rows':0}
    if len(groups) != len(set(groups)):
        raise ValueError('Augmentation is restricted to one-row-per-person representations.')
    rng = np.random.default_rng(seed)
    xx, yy, gg, parents = [X], [y], [groups], []
    counts = np.bincount(y)
    for cls in range(len(counts)):
        ix = np.flatnonzero(y==cls)
        count = int(counts.max()-len(ix))
        if not count:
            continue
        if method == 'smote' and len(ix)>=2:
            neighbors = NearestNeighbors(n_neighbors=min(6,len(ix))).fit(X[ix]).kneighbors(X[ix],return_distance=False)
        for k in range(count):
            a = int(rng.integers(len(ix)))
            first = ix[a]
            if method == 'smote' and len(ix)>=2:
                options = neighbors[a][neighbors[a] != a]
                b = int(rng.choice(options)) if len(options) else a
                second = ix[b]
                row = X[first] + rng.uniform()*(X[second]-X[first])
            else:
                second = first
                row = X[first] + rng.normal(0,.03,X.shape[1])
            xx.append(row[None,:]); yy.append(np.array([cls]))
            gg.append(np.array([f'synthetic_{cls}_{k}']))
            parents.append([str(groups[first]),str(groups[second])])
    assert all(set(p).issubset(set(groups)) for p in parents)
    return np.concatenate(xx),np.concatenate(yy),np.concatenate(gg),{
        'method':method,'synthetic_rows':len(parents),'parents':parents}


def model_scores(model, X, family, n_classes):
    if family == 'rbf_ridge':
        scores = softmax(model.predict(X)*3,axis=1)
    elif hasattr(model,'predict_proba'):
        scores = model.predict_proba(X)
    else:
        decision = model.decision_function(X)
        if decision.ndim == 1:
            p = expit(decision)
            scores = np.column_stack([1-p,p])
        else:
            scores = softmax(decision,axis=1)
    if scores.shape != (len(X),n_classes) or not np.isfinite(scores).all():
        raise ValueError('Invalid estimator scores.')
    scores = np.clip(scores,1e-12,1)
    return scores/scores.sum(axis=1,keepdims=True)


@dataclass
class Fitted:
    spec: Spec
    processor: Processor
    model: object
    n_classes: int
    training_subjects: tuple
    augmentation_audit: dict

    def predict(self, representations, subjects):
        rep = representations[self.spec.representation]
        indices = rep.rows_for(subjects)
        p = model_scores(self.model,self.processor.transform(rep.X.iloc[indices]),self.spec.family,self.n_classes)
        if self.spec.pooling == 'geometric':
            p = np.log(p)
        pooled = pd.DataFrame(p,index=rep.groups[indices]).groupby(level=0).mean().reindex(subjects).to_numpy()
        if self.spec.pooling == 'geometric':
            pooled = np.exp(pooled)
        if not np.isfinite(pooled).all():
            raise ValueError('Missing subject prediction.')
        return pooled/pooled.sum(axis=1,keepdims=True)


def fit_candidate(training, spec, threads=4):
    labels = task_labels(training.labels,spec.task)
    rep = training.representations[spec.representation]
    indices = rep.rows_for(labels.index)
    groups = rep.groups[indices]
    y = labels.reindex(groups).to_numpy(int)
    classes = len(np.unique(y))
    assert set(y) == set(range(classes)) and classes == (3 if spec.task=='flat' else 2)
    processor = Processor(spec).fit(rep.X.iloc[indices],y,groups,rep.meta)
    X = processor.transform(rep.X.iloc[indices])
    X,y,fit_groups,audit = augment_training(X,y,groups,spec.augmentation,spec.seed)
    model = estimator(spec,classes,threads)
    if spec.family == 'lda':
        if len(groups) != len(set(groups)):
            raise ValueError('LDA does not support repeated-window subject weighting.')
        prior = np.bincount(y).astype(float)**(1-spec.balance)
        model.set_params(priors=prior/prior.sum()).fit(X,y)
    elif spec.family == 'rbf_ridge':
        model.fit(X,np.eye(classes)[y],sample_weight=subject_weights(fit_groups,y,spec.balance))
    else:
        model.fit(X,y,sample_weight=subject_weights(fit_groups,y,spec.balance))
    return Fitted(spec,processor,model,classes,tuple(sorted(set(groups))),audit)
