"""Validation-driven model selection, explicitly not unseen-test estimation."""
from __future__ import annotations
from dataclasses import dataclass
import itertools

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, roc_curve

from lifelog_v2.metrics import measure, target_status


def normalize(p):
    p = np.clip(np.asarray(p,dtype=float),1e-12,None)
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite scores.')
    return p/p.sum(axis=1,keepdims=True)


def decision_policy(y,p):
    """Tune two global class offsets on the declared selection set only."""
    logits = np.log(np.clip(p,1e-12,1))
    truth = np.asarray(y,int)
    actual = np.bincount(truth,minlength=3)
    if (actual==0).any():
        raise ValueError('The selection set must contain all three classes.')
    center = np.zeros(2)
    best = None
    for grid in [np.linspace(-4,4,25), np.linspace(-.35,.35,15)]:
        offsets = np.array([(0,a+center[0],b+center[1]) for a,b in itertools.product(grid,grid)])
        if best is not None:
            offsets = np.vstack([offsets,best[1]])
        pred = (logits[None,:,:]+offsets[:,None,:]).argmax(axis=2)
        tp = np.column_stack([((pred==k)&(truth[None,:]==k)).sum(axis=1) for k in range(3)])
        pc = np.column_stack([(pred==k).sum(axis=1) for k in range(3)])
        recalls = tp/actual
        macro = recalls.mean(axis=1)
        minimum = recalls.min(axis=1)
        f1 = (2*tp/(pc+actual)).mean(axis=1)
        norm = np.linalg.norm(offsets-offsets.mean(axis=1,keepdims=True),axis=1)
        winner = np.lexsort((-norm,f1,minimum,macro))[-1]
        best = (pred[winner],offsets[winner])
        center = offsets[winner,1:].copy()
    prediction, offset = best
    offset = offset-offset.mean()
    return offset,measure(truth,p,prediction)


def objective(metrics,threshold=.8):
    a,r = metrics['roc_auc_macro_ovr'],metrics['macro_recall']
    # Crossing BOTH requested endpoints takes priority, then optimize the weaker.
    return float((a>=threshold and r>=threshold) + min(a,r) + .02*(a+r)/2 + .001*metrics['macro_f1'])


def refine_policy(y,p,initial):
    """Search decision changes between score breakpoints, beyond a fixed grid.

    For a fixed MCI offset, DEM decisions change at max(CN,MCI)-DEM.
    These thresholds reorder when a CN line crosses an MCI line. Enumerate
    intervals between those crossings; retain the incoming policy as a candidate.
    Only two global offsets are tuned, never subject-specific decisions.
    """
    logits=np.log(np.clip(np.asarray(p,float),1e-12,1))
    truth=np.asarray(y,int)
    actual=np.bincount(truth,minlength=3)
    if (actual==0).any(): raise ValueError('All three selection classes are required.')
    def intervals(values):
        values=np.unique(values)
        return np.r_[values[0]-1,(values[:-1]+values[1:])/2,values[-1]+1]
    cn=logits[:,0]-logits[:,2]
    mci=logits[:,1]-logits[:,2]
    crossings=(cn[:,None]-mci[None,:]).ravel()
    candidates=[np.asarray(initial,float)-initial[0]]
    for a in intervals(crossings):
        b=intervals(np.maximum(cn,mci+a))
        candidates.extend(np.column_stack([np.zeros(len(b)),np.full(len(b),a),b]))
    offsets=np.asarray(candidates)
    pred=(logits[None,:,:]+offsets[:,None,:]).argmax(axis=2)
    tp=np.column_stack([((pred==k)&(truth[None,:]==k)).sum(axis=1) for k in range(3)])
    pc=np.column_stack([(pred==k).sum(axis=1) for k in range(3)])
    recalls=tp/actual
    f1=(2*tp/(pc+actual)).mean(axis=1)
    norm=np.linalg.norm(offsets-offsets.mean(axis=1,keepdims=True),axis=1)
    winner=np.lexsort((-norm,f1,recalls.min(axis=1),recalls.mean(axis=1)))[-1]
    # Preserve the actual evaluated offsets, including tie behavior.
    return offsets[winner],measure(truth,p,pred[winner])


def binary_metrics(labels,p,task):
    if task=='cn_mci':
        mask = labels.to_numpy()!=2
        y = labels.to_numpy()[mask]
        score = p[mask,1]
    else:
        y = labels.eq(2).to_numpy(int)
        score = p[:,1]
    auc = roc_auc_score(y,score)
    fpr,tpr,_ = roc_curve(y,score)
    balanced = np.max((tpr+1-fpr)/2)
    return {'binary_auc':float(auc),'binary_best_balanced_recall_selection':float(balanced)}, float(auc+.02*balanced)


def recipe_scores(recipe,predictions):
    kind = recipe['kind']
    if kind=='flat':
        p = predictions[recipe['id']]
        return normalize(np.exp(np.log(np.clip(p,1e-12,1))/recipe.get('temperature',1.) + np.array(recipe.get('bias',[0,0,0]))))
    if kind=='hierarchy':
        d = np.clip(predictions[recipe['dem_id']][:,1],1e-9,1-1e-9)
        q = np.clip(predictions[recipe['cn_mci_id']][:,1],1e-9,1-1e-9)
        d = expit(logit(d)/recipe.get('dem_temperature',1.) + recipe.get('dem_bias',0.))
        q = expit(logit(q)/recipe.get('cn_mci_temperature',1.) + recipe.get('cn_mci_bias',0.))
        return normalize(np.column_stack([(1-d)*(1-q),(1-d)*q,d]))
    if kind=='ensemble':
        matrices = np.stack([recipe_scores(r,predictions) for r in recipe['members']])
        weights = np.asarray(recipe['weights'],float)
        if weights.ndim==1:
            weights = np.repeat(weights[:,None],3,axis=1)
        weights = weights/weights.sum(axis=0,keepdims=True)
        return normalize((matrices*weights[:,None,:]).sum(axis=0))
    raise ValueError(kind)


def recipe_ids(recipe):
    if recipe['kind']=='flat':
        return {recipe['id']}
    if recipe['kind']=='hierarchy':
        return {recipe['dem_id'],recipe['cn_mci_id']}
    return set().union(*(recipe_ids(r) for r in recipe['members']))


@dataclass
class Champion:
    models: dict
    recipe: dict
    offsets: np.ndarray
    protocol: dict

    def predict(self,representations,subjects):
        predictions = {key:m.predict(representations,subjects) for key,m in self.models.items()}
        p = recipe_scores(self.recipe,predictions)
        h = (np.log(p)+self.offsets).argmax(axis=1)
        return p,h

    def predict_csv(self,activity_csv,sleep_csv):
        from .data import inference_data
        representations,subjects = inference_data(activity_csv,sleep_csv)
        p,h = self.predict(representations,subjects)
        out = pd.DataFrame(p,index=subjects,columns=['score_CN','score_MCI','score_DEM'])
        out.index.name = 'subject_hash'
        out['predicted'] = np.array(['CN','MCI','DEM'])[h]
        return out
