"""Contract and pipeline checks with real classifier fitting explicitly blocked.

Synthetic fixtures are kept in TemporaryDirectory and never mixed with results.
The spy's fit method records routing only; it does not learn parameters.
"""
from __future__ import annotations
import contextlib
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from lifelog_v2.data import Dataset,Representation
from .data import temporal_summary,hr_features
from .models import Spec,Processor,augment_training,task_labels,estimator,families,fit_candidate
from .selection import Champion,decision_policy,refine_policy,recipe_scores
from .search import Config,run


class FitSpy:
    seen=[]

    def fit(self,X,y,sample_weight=None):
        type(self).seen.append({'n':len(X),'y':np.array(y).copy()})
        self.classes_=np.unique(y)
        return self

    def predict_proba(self,X):
        score=np.column_stack([np.exp(np.clip(X[:,0]*(k+1)/5,-5,5)) for k in self.classes_])
        return score/score.sum(axis=1,keepdims=True)


def fixture(prefix,n):
    rng=np.random.default_rng(190 if prefix=='train' else 191)
    subjects=np.array([f'{prefix}_{i:03}' for i in range(n)])
    columns=['activity__low__extra_std','sleep__total__median','v3__sleep__paper_hr_drop_ratio__std']
    X=pd.DataFrame(rng.normal(size=(n,3)),index=subjects,columns=columns)
    labels=pd.Series(np.arange(n)%3,index=subjects)
    meta=pd.DataFrame({'modality':['activity','sleep','sleep'],'family':['physiology']*3},index=columns)
    reps={'subject':Representation(X,subjects,meta)}
    return Dataset(labels,reps,[],{'fixture':prefix})


@contextlib.contextmanager
def classifier_guard():
    from . import models
    classes=[models.LogisticRegression,models.SVC,models.CatBoostClassifier,
             models.LinearDiscriminantAnalysis,models.ExtraTreesClassifier,
             models.RandomForestClassifier,models.HistGradientBoostingClassifier,models.KernelRidge]
    if 'xgboost' in families():
        from xgboost import XGBClassifier
        classes.append(XGBClassifier)
    if 'lightgbm' in families():
        from lightgbm import LGBMClassifier
        classes.append(LGBMClassifier)
    with contextlib.ExitStack() as stack:
        for cls in classes:
            stack.enter_context(patch.object(cls,'fit',side_effect=AssertionError('Real classifier training is forbidden in these checks.')))
        yield


class Contracts(unittest.TestCase):
    def test_temporal_gaps(self):
        x=pd.Series([10.,11.,100.,101.],index=pd.to_datetime(['2026-01-01','2026-01-02','2026-01-05','2026-01-06']))
        m=temporal_summary(x)
        self.assertEqual(m['moving_range'],1.)
        self.assertEqual(m['rmssd'],1.)

    def test_hr_positions(self):
        row={'CONVERT(sleep_hr_5min USING utf8)':'0/100/90/80/70',
             'sleep_deep':3600,'sleep_light':3600,'sleep_total':10800}
        m=hr_features(row)
        self.assertEqual(m['sleep__paper_hr_baseline_first15min'],95.)
        self.assertAlmostEqual(m['sleep__paper_hr_drop_ratio'],25/95)
        self.assertAlmostEqual(m['sleep__paper_hr_drop_firstvalid_ratio'],20/90)

    def test_task_membership(self):
        ds=fixture('train',18)
        self.assertEqual(len(task_labels(ds.labels,'dem')),18)
        self.assertEqual(len(task_labels(ds.labels,'cn_mci')),12)
        self.assertTrue(set(task_labels(ds.labels,'cn_mci').index).isdisjoint(ds.labels[ds.labels.eq(2)].index))

    def test_processor_train_only(self):
        ds=fixture('train',18)
        p=Processor(Spec(top_k=2)).fit(ds.representations['subject'].X,ds.y,ds.subjects,ds.representations['subject'].meta)
        before=p.medians.copy()
        transformed=p.transform(fixture('selection',9).representations['subject'].X*1e6)
        self.assertTrue(np.isfinite(transformed).all())
        pd.testing.assert_series_equal(before,p.medians)

    def test_augmentation_parentage(self):
        groups=np.array(['a','b','c','d','e','f'])
        X=np.arange(18,dtype=float).reshape(6,3)
        y=np.array([0,0,0,0,1,1])
        xx,yy,gg,audit=augment_training(X,y,groups,'smote',2026)
        self.assertEqual(len(xx),8)
        self.assertTrue(all(set(parents)<=set(groups) for parents in audit['parents']))
        self.assertEqual(np.bincount(yy).tolist(),[4,4])

    def test_factory_apis_without_fit(self):
        # Constructors must work while every native classifier fit is blocked.
        for family in families():
            for n in [2,3]:
                m=estimator(Spec(family=family),n,1)
                self.assertTrue(callable(m.fit))
                self.assertTrue(callable(m.get_params))

    def test_hierarchy_and_policy(self):
        predictions={'d':np.tile([.8,.2],(6,1)),'q':np.tile([.6,.4],(6,1))}
        r={'kind':'hierarchy','dem_id':'d','cn_mci_id':'q'}
        p=recipe_scores(r,predictions)
        np.testing.assert_allclose(p,np.tile([.48,.32,.2],(6,1)))
        y=np.array([0,1,2,0,1,2])
        offsets,metrics=decision_policy(y,p)
        self.assertAlmostEqual(metrics['roc_auc_macro_ovr'],.5)
        self.assertGreaterEqual(metrics['macro_recall'],1/3)
        self.assertTrue(np.isfinite(offsets).all())

    def test_pipeline_with_training_spy(self):
        training,selection=fixture('train',18),fixture('selection',9)
        specs=[Spec(task=task,family='logreg',top_k=2) for task in ['flat','dem','cn_mci']]
        FitSpy.seen=[]
        with tempfile.TemporaryDirectory(prefix='lifelog-v3-contract-') as tmp:
            cfg=Config(flat_trials=0,dem_trials=0,cn_mci_trials=0,recipe_trials=2,ensemble_trials=2,
                       top_branch=2,threads=1,output_dir=tmp)
            with patch('lifelog_v3.search.load_split',side_effect=[training,selection]), \
                 patch('lifelog_v3.search.baseline_specs',return_value=specs), \
                 patch('lifelog_v3.search.families',return_value=['logreg']), \
                 patch('lifelog_v3.models.estimator',side_effect=lambda *a,**k:FitSpy()), \
                 contextlib.redirect_stdout(io.StringIO()):
                result=run(cfg)
            self.assertEqual(result['subject_overlap'],0)
            self.assertEqual(result['successful_unique_candidates'],3)
            self.assertTrue(all(r['n'] in [12,18] for r in FitSpy.seen))
            champion=joblib.load(Path(tmp)/'champion_selection_only.joblib')
            p,h=champion.predict(selection.representations,selection.subjects)
            self.assertEqual(p.shape,(9,3));self.assertEqual(h.shape,(9,))
            self.assertTrue(all(set(m.training_subjects).isdisjoint(selection.subjects) for m in champion.models.values()))
            self.assertTrue((Path(tmp)/'RESULTS.md').exists())

    def test_policy_refinement(self):
        rng=np.random.default_rng(27)
        p=rng.dirichlet([1,1,1],18)
        y=np.arange(18)%3
        offsets,before=decision_policy(y,p)
        final,after=refine_policy(y,p,offsets)
        self.assertGreaterEqual(after['macro_recall'],before['macro_recall'])
        self.assertEqual(after['roc_auc_macro_ovr'],before['roc_auc_macro_ovr'])
        predicted=(np.log(p)+final).argmax(axis=1)
        recall=np.mean([np.mean(predicted[y==k]==k) for k in range(3)])
        self.assertAlmostEqual(recall,after['macro_recall'])


def main():
    with classifier_guard():
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
        result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'real_classifier_fits':0,'pipeline_fixture':'synthetic, temporary, estimator replaced by routing spy',
            'status':'passed' if result.wasSuccessful() else 'failed'}
    Path('results_v3').mkdir(exist_ok=True)
    Path('results_v3/contract_checks.json').write_text(json.dumps(report,indent=2))
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__=='__main__':
    main()
