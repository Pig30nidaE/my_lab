"""Tiny synthetic-only native-library checks, never load wearable data.

No accuracy or AUC from this artificial fixture is a project result.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from lifelog_v2.data import Dataset,Representation
from .models import Spec,families,fit_candidate


def fixture(prefix,n):
    rng=np.random.default_rng(72 if prefix=='train' else 18)
    subjects=np.array([f'synthetic_{prefix}_{i}' for i in range(n)])
    columns=[f'activity__synthetic_{i}__std' for i in range(12)]
    values=rng.normal(size=(n,12))
    labels=pd.Series(np.arange(n)%3,index=subjects)
    values[:,0]+=labels.to_numpy()
    values[::9,3]=np.nan
    frame=pd.DataFrame(values,index=subjects,columns=columns)
    meta=pd.DataFrame({'modality':'activity','family':'physiology'},index=columns)
    return Dataset(labels,{'subject':Representation(frame,subjects,meta)},[],{'synthetic':prefix})


def main():
    training,selection=fixture('train',72),fixture('selection',18)
    cases=[]
    for task in ['flat','dem','cn_mci']:
        for i,family in enumerate(families()):
            params=({'iterations':3} if family=='catboost' else
                    {'n_estimators':3} if family in ['randomforest','extratrees','xgboost','lightgbm'] else
                    {'max_iter':3} if family=='histgb' else {})
            cases.append(Spec(task=task,family=family,top_k=7,params=params,
                              transform=['standard','robust','signedlog','quantile'][i%4],
                              augmentation='smote' if task=='dem' else 'none',
                              pca=3 if family in ['lda','svm'] else 0))
    cases.append(Spec(task='flat',family='logreg',top_k=3,selection='rfe'))
    completed=[]
    with tempfile.TemporaryDirectory(prefix='v3-native-smoke-') as tmp, threadpool_limits(limits=1):
        for spec in cases:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='X does not have valid feature names')
                model=fit_candidate(training,spec,threads=1)
                scores=model.predict(selection.representations,selection.subjects)
                assert scores.shape==(18,3 if spec.task=='flat' else 2)
                np.testing.assert_allclose(scores.sum(axis=1),1.,atol=1e-6)
                assert set(model.training_subjects).isdisjoint(selection.subjects)
                path=Path(tmp)/'model.joblib'
                joblib.dump(model,path)
                restored=joblib.load(path)
                np.testing.assert_allclose(scores,restored.predict(selection.representations,selection.subjects))
            completed.append({'task':spec.task,'family':spec.family,'selection':spec.selection})
    report={'status':'passed','real_data_classifier_fits':0,
            'synthetic_native_candidates':len(completed),'synthetic_subjects_train':72,
            'synthetic_subjects_prediction':18,'cases':completed,
            'note':'API, preprocessing, probability shape and serialization checks only; no performance claim.'}
    Path('results_v3').mkdir(exist_ok=True)
    Path('results_v3/native_smoke.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
