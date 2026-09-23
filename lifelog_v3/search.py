"""Single fixed subject split. No LLM/API calls and no nested CV.

Validation labels are deliberately used for hyperparameter/mixture/threshold
selection. All exported performance is SELECTION performance, not an independent
generalization estimate. Estimator/preprocessor fitting never sees these people.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict,replace
from pathlib import Path
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import time
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from threadpoolctl import threadpool_limits

from lifelog_v2 import feature_engineering as fe
from lifelog_v2.metrics import measure,target_status
from .data import load_split
from .models import Spec,families,fit_candidate,view_columns
from .selection import Champion,decision_policy,refine_policy,objective,binary_metrics,recipe_scores,recipe_ids


@dataclass
class Config:
    seed: int = 2026
    threads: int = 4
    flat_trials: int = 160
    dem_trials: int = 100
    cn_mci_trials: int = 160
    recipe_trials: int = 2500
    ensemble_trials: int = 1200
    top_branch: int = 10
    threshold: float = .8
    output_dir: str = 'results_v3/run_full'


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def source_hashes():
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ['lifelog_v3','lifelog_v2']
            for p in sorted(Path(directory).glob('*.py'))}


def package_versions():
    names=['numpy','pandas','scipy','scikit-learn','catboost','optuna','joblib']
    result={p:importlib.metadata.version(p) for p in names}
    for p in ['xgboost','lightgbm']:
        try: result[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: result[p]='not installed'
    return result


def controlled_ablation_specs():
    reference=Spec(task='flat',family='catboost',view='no_context',top_k=40)
    changes={
        'reference':{}, 'add_collection_context':{'view':'all'},
        'context_only':{'view':'context_only'}, 'activity_only':{'view':'activity'},
        'sleep_only':{'view':'sleep'}, 'legacy_only':{'view':'legacy'},
        'temporal_only':{'view':'temporal'}, 'no_class_balance':{'balance':0.},
        'sqrt_class_balance':{'balance':.5}, 'smote':{'augmentation':'smote'},
        'jitter':{'augmentation':'jitter'}, 'standard_scale':{'transform':'standard'},
        'mutual_information':{'selection':'mutual_info'}, 'rfe':{'selection':'rfe'},
        'top10':{'top_k':10}, 'top160':{'top_k':160}}
    return {name:replace(reference,**change) for name,change in changes.items()}


def baseline_specs():
    out=[]
    for task in ['flat','dem','cn_mci']:
        views = ['legacy','temporal','compact','circadian'] if task=='flat' else (
            ['dem_single','activity_variability','compact'] if task=='dem' else
            ['paper_sleep','hr_drop','variability','compact'])
        for view in views:
            for family in ['logreg','lda','catboost']:
                out.append(Spec(task=task,family=family,view=view,top_k='all' if view in ['dem_single','compact'] else 20))
        if task=='cn_mci':
            out.extend([Spec(task=task,family='logreg',view='paper_sleep',top_k=k,selection='rfe') for k in [5,10,40]])
    out.extend([Spec(task='flat',family=f,representation=r,view='all',top_k=80)
                for r in ['window14','window28'] for f in ['catboost','logreg']])
    out.extend([Spec(task='flat',family='catboost',view=view,top_k=40)
                for view in ['all','no_context','context_only']])
    out.extend(controlled_ablation_specs().values())
    return list({digest(spec.dictionary()):spec for spec in out}.values())


def suggest(trial,task,family,seed):
    view = trial.suggest_categorical('view',['all','no_context','legacy','temporal','variability','activity_variability','activity','sleep','physiology','paper_sleep','hr_drop','circadian','compact'] + (['dem_single'] if task=='dem' else []))
    spec=Spec(task=task,family=family,view=view,seed=seed,
              top_k=trial.suggest_categorical('top_k',[1,3,5,10,20,40,80,160,'all']),
              selection=trial.suggest_categorical('selection',['anova','mutual_info','rfe']),
              transform=trial.suggest_categorical('transform',['standard','robust','signedlog','quantile']),
              correlation=trial.suggest_categorical('correlation',[.9,.98,1.]),
              balance=trial.suggest_categorical('balance',[0.,.5,1.]),
              augmentation=trial.suggest_categorical('augmentation',['none','smote','jitter']))
    if spec.selection=='rfe' and spec.top_k=='all':
        spec.top_k=80
    if view in ['dem_single','compact']:
        spec.selection='anova'; spec.correlation=1.; spec.top_k='all'
    if family in ['logreg','svm','lda','rbf_ridge']:
        spec.pca=trial.suggest_categorical('pca',[0,5,15,30])
    if family=='logreg':
        spec.params={'C':trial.suggest_float('C',1e-4,100,log=True)}
    elif family=='svm':
        spec.params={'C':trial.suggest_float('C',.01,100,log=True),
                     'gamma':trial.suggest_categorical('gamma',['scale',.0001,.001,.01,.1,1.])}
    elif family=='lda':
        spec.params={'shrinkage':trial.suggest_categorical('shrinkage',['auto',.1,.3,.6,.9])}
    elif family=='rbf_ridge':
        spec.params={'alpha':trial.suggest_float('alpha',.001,100,log=True),
                     'gamma':trial.suggest_float('gamma',1e-5,1.,log=True)}
    elif family in ['randomforest','extratrees']:
        spec.params={'n_estimators':500,'max_depth':trial.suggest_categorical('depth',[3,5,8,None]),
                     'min_samples_leaf':trial.suggest_int('leaf',1,10),
                     'max_features':trial.suggest_categorical('features',['sqrt',.3,.7,1.])}
    elif family=='catboost':
        spec.params={'iterations':trial.suggest_categorical('iterations',[300,600,1000]),
                     'depth':trial.suggest_int('depth',2,6),'learning_rate':trial.suggest_float('lr',.015,.12,log=True),
                     'l2_leaf_reg':trial.suggest_float('l2',1,100,log=True),
                     'boosting_type':trial.suggest_categorical('boosting',['Plain','Ordered'])}
    elif family=='histgb':
        spec.params={'max_iter':trial.suggest_categorical('iterations',[150,350,650]),
                     'max_leaf_nodes':trial.suggest_categorical('leaves',[3,7,15]),
                     'min_samples_leaf':trial.suggest_int('leaf',3,20),
                     'learning_rate':trial.suggest_float('lr',.02,.15,log=True),
                     'l2_regularization':trial.suggest_float('l2',.1,100,log=True)}
    elif family=='xgboost':
        spec.params={'n_estimators':trial.suggest_categorical('iterations',[300,600,1000]),
                     'max_depth':trial.suggest_int('depth',2,5),'learning_rate':trial.suggest_float('lr',.01,.15,log=True),
                     'reg_lambda':trial.suggest_float('l2',.5,100,log=True),
                     'min_child_weight':trial.suggest_float('child',.5,10,log=True)}
    elif family=='lightgbm':
        spec.params={'n_estimators':trial.suggest_categorical('iterations',[300,600,1000]),
                     'num_leaves':trial.suggest_categorical('leaves',[3,7,15]),
                     'learning_rate':trial.suggest_float('lr',.01,.15,log=True),
                     'reg_lambda':trial.suggest_float('l2',.1,100,log=True),
                     'min_child_samples':trial.suggest_int('leaf',3,20)}
    return spec


class Search:
    def __init__(self,training,validation,config,output):
        self.training,self.validation,self.config,self.output=training,validation,config,Path(output)
        if set(training.subjects).intersection(validation.subjects):
            raise ValueError('Train/selection subject overlap.')
        self.records={}; self.predictions={}
        failure_path=self.output/'failures.json'
        self.failures=json.loads(failure_path.read_text()) if failure_path.exists() else []
        (self.output/'candidates').mkdir(exist_ok=True,parents=True)

    def evaluate(self,spec):
        key=digest(spec.dictionary())[:24]
        if key in self.records:
            return self.records[key]
        path=self.output/'candidates'/key
        started=time.monotonic()
        if path.with_suffix('.json').exists() and path.with_suffix('.npz').exists():
            record=json.loads(path.with_suffix('.json').read_text())
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
                scores=z['scores']
        else:
            with threadpool_limits(limits=self.config.threads):
                fitted=fit_candidate(self.training,spec,self.config.threads)
                assert not set(fitted.training_subjects).intersection(self.validation.subjects)
                scores=fitted.predict(self.validation.representations,self.validation.subjects)
            if spec.task=='flat':
                offsets,metrics=decision_policy(self.validation.y,scores)
                value=objective(metrics,self.config.threshold)
                raw=measure(self.validation.y,scores)
            else:
                metrics,value=binary_metrics(self.validation.labels,scores,spec.task)
                offsets=np.zeros(3);raw={}
            record={'id':key,'spec':spec.dictionary(),'selection_metrics':metrics,'raw_metrics':raw,
                    'objective':value,'offsets':offsets.tolist(),'training_subjects':len(fitted.training_subjects),
                    'selected_features':list(fitted.processor.columns),'synthetic_rows':fitted.augmentation_audit['synthetic_rows'],
                    'seconds':time.monotonic()-started}
            temporary=path.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream: np.savez_compressed(stream,scores=scores)
            temporary.replace(path.with_suffix('.npz'))
            write_json(path.with_suffix('.json'),record)
        self.records[key]=record; self.predictions[key]=scores
        self.save_table()
        return record

    def save_table(self):
        rows=[dict(id=r['id'],task=r['spec']['task'],family=r['spec']['family'],view=r['spec']['view'],
                   representation=r['spec']['representation'],top_k=r['spec']['top_k'],selection=r['spec']['selection'],
                   augmentation=r['spec']['augmentation'],objective=r['objective'],seconds=r['seconds'],
                   **r['selection_metrics']) for r in self.records.values()]
        pd.DataFrame(rows).to_csv(self.output/'candidate_selection_scores.csv',index=False)

    def failure(self,stage,exc,spec=None):
        self.failures.append({'stage':stage,'error':f'{type(exc).__name__}: {exc}',
                              'spec':None if spec is None else spec.dictionary()})
        write_json(self.output/'failures.json',self.failures)
        print(f'Candidate failed at {stage}: {type(exc).__name__}: {exc}',flush=True)

    def fit_bank(self):
        # Baselines are interleaved across tasks; family budgets cannot collapse
        # into the same LDA plateau as a single joint-family TPE search can.
        for i,spec in enumerate(baseline_specs()):
            print(f'Baseline {i+1}/{len(baseline_specs())}: {spec.task}/{spec.family}/{spec.view}',flush=True)
            try: self.evaluate(spec)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc: self.failure('baseline',exc,spec)
        available=families()
        for task in ['flat','dem','cn_mci']:
            total=getattr(self.config,f'{task}_trials')
            for f,family in enumerate(available):
                budget=total//len(available)+(f<total%len(available))
                study_name=f'{task}_{family}'
                study=optuna.create_study(direction='maximize',study_name=study_name,
                    storage=f"sqlite:///{(self.output/'studies.sqlite3').resolve()}",load_if_exists=True,
                    sampler=optuna.samplers.TPESampler(seed=self.config.seed+f,n_startup_trials=min(10,max(3,budget//3))))
                # Output folder is a single-process run. Incomplete trials from
                # a prior interrupted process are recorded as failed on resume.
                for old in study.trials:
                    if old.state==optuna.trial.TrialState.RUNNING:
                        study.tell(old.number,state=optuna.trial.TrialState.FAIL)
                    if old.state==optuna.trial.TrialState.COMPLETE:
                        self.evaluate(Spec(**old.user_attrs['spec']))
                def run_trial(trial):
                    spec=suggest(trial,task,family,self.config.seed)
                    trial.set_user_attr('spec',spec.dictionary())
                    try:
                        record=self.evaluate(spec)
                        trial.set_user_attr('candidate_id',record['id'])
                        return record['objective']
                    except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                        self.failure(study_name,exc,spec)
                        raise
                count=max(0,budget-len(study.trials))
                print(f'Search {study_name}: {count} remaining trials',flush=True)
                study.optimize(run_trial,n_trials=count,n_jobs=1,catch=(ValueError,RuntimeError,np.linalg.LinAlgError))
                study.trials_dataframe().to_csv(self.output/f'trials_{study_name}.csv',index=False)
        for task in ['flat','dem','cn_mci']:
            if not any(r['spec']['task']==task for r in self.records.values()):
                raise RuntimeError(f'No successful candidates for {task}; inspect failures.json.')

    def candidates(self,task,n):
        records=[r for r in self.records.values() if r['spec']['task']==task]
        records.sort(key=lambda r:r['objective'],reverse=True)
        # Remove predictions that are numerically identical, not just parameter duplicates.
        chosen=[]
        for r in records:
            if not any(np.allclose(self.predictions[r['id']],self.predictions[c['id']],atol=1e-8) for c in chosen):
                chosen.append(r)
            if len(chosen)>=n: break
        return chosen

    def select_recipe(self):
        flat=self.candidates('flat',20)
        dem=self.candidates('dem',self.config.top_branch)
        cn_mci=self.candidates('cn_mci',self.config.top_branch)
        evaluated={}; traces=[]
        def evaluate(recipe,stage):
            key=digest(recipe)
            if key in evaluated: return evaluated[key]
            p=recipe_scores(recipe,self.predictions)
            offsets,metrics=decision_policy(self.validation.y,p)
            result={'recipe':recipe,'metrics':metrics,'offsets':offsets.tolist(),
                    'objective':objective(metrics,self.config.threshold)}
            evaluated[key]=result
            traces.append(dict(stage=stage,recipe_id=key,objective=result['objective'],**metrics))
            if len(evaluated)==1 or result['objective']>max(r['objective'] for k,r in evaluated.items() if k!=key):
                write_json(self.output/'best_recipe_so_far.json',result)
            return result
        for r in flat: evaluate({'kind':'flat','id':r['id']},'direct_3class')
        for d in dem:
            for c in cn_mci:
                evaluate({'kind':'hierarchy','dem_id':d['id'],'cn_mci_id':c['id']},'hierarchy')
        rng=np.random.default_rng(self.config.seed)
        for i in range(self.config.recipe_trials):
            if i%4==0:
                r=flat[int(rng.integers(len(flat)))]
                recipe={'kind':'flat','id':r['id'],'temperature':float(np.exp(rng.uniform(-1.2,1.2))),
                        'bias':[0.,float(rng.uniform(-1.2,1.2)),float(rng.uniform(-1.2,1.2))]}
            else:
                d,c=dem[int(rng.integers(len(dem)))],cn_mci[int(rng.integers(len(cn_mci)))]
                recipe={'kind':'hierarchy','dem_id':d['id'],'cn_mci_id':c['id'],
                        'dem_temperature':float(np.exp(rng.uniform(-1.2,1.2))),
                        'cn_mci_temperature':float(np.exp(rng.uniform(-1.2,1.2))),
                        'dem_bias':float(rng.uniform(-1.5,1.5)),'cn_mci_bias':float(rng.uniform(-1.5,1.5))}
            evaluate(recipe,'score_transform')
        ranked=sorted(evaluated.values(),key=lambda r:r['objective'],reverse=True)
        pool=[]
        seen=set()
        for r in ranked:
            signature=tuple(sorted(recipe_ids(r['recipe'])))
            if signature not in seen:
                pool.append(r['recipe']);seen.add(signature)
            if len(pool)>=20: break
        # Unconstrained learned stacking on selection subjects is deliberately
        # absent. These are finite global voting-weight hyperparameter searches.
        for i in range(self.config.ensemble_trials):
            size=int(rng.integers(2,min(5,len(pool))+1)) if len(pool)>1 else 1
            indexes=rng.choice(len(pool),size=size,replace=False)
            members=[pool[int(j)] for j in indexes]
            if i%2:
                weights=np.column_stack([rng.dirichlet(np.ones(size)) for _ in range(3)])
            else:
                weights=rng.dirichlet(np.ones(size))
            evaluate({'kind':'ensemble','members':members,'weights':weights.tolist()},'soft_vote')
        ranked=sorted(evaluated.values(),key=lambda r:r['objective'],reverse=True)
        auc_ranked=sorted(evaluated.values(),key=lambda r:r['metrics']['roc_auc_macro_ovr'],reverse=True)
        refinements={digest(r['recipe']):r for r in ranked[:64]+auc_ranked[:32]}
        print(f'Refining global decision offsets for {len(refinements)} top score combinations',flush=True)
        for key,result in refinements.items():
            p=recipe_scores(result['recipe'],self.predictions)
            offsets,metrics=refine_policy(self.validation.y,p,np.array(result['offsets']))
            result.update(metrics=metrics,offsets=offsets.tolist(),objective=objective(metrics,self.config.threshold))
            traces.append(dict(stage='offset_refinement',recipe_id=key,objective=result['objective'],**metrics))
        pd.DataFrame(traces).to_csv(self.output/'recipe_selection_scores.csv',index=False)
        winner=max(evaluated.values(),key=lambda r:r['objective'])
        write_json(self.output/'selected_recipe.json',winner)
        return winner


@contextlib.contextmanager
def run_lock(output):
    """Prevent two kernels from resuming/writing the same study concurrently."""
    stream=(Path(output)/'.run.lock').open('a+b')
    try:
        if __import__('os').name=='nt':
            import msvcrt
            stream.write(b'0');stream.flush();stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError('Another process is already using this output_dir.') from exc
    try:
        yield
    finally:
        stream.close()


def run(config=None,data_root=None):
    config=Config() if config is None else config
    output=Path(config.output_dir);output.mkdir(parents=True,exist_ok=True)
    with run_lock(output):
        return _run(config,data_root)


def _run(config=None,data_root=None):
    config=Config() if config is None else config
    if min(config.flat_trials,config.dem_trials,config.cn_mci_trials,config.recipe_trials,config.ensemble_trials)<0:
        raise ValueError('Search budgets cannot be negative.')
    if config.threads<1 or config.top_branch<1: raise ValueError('Invalid resource/bank limits.')
    output=Path(config.output_dir);output.mkdir(parents=True,exist_ok=True)
    _,paths=fe.resolve_paths(data_root)
    training,validation=load_split(paths['train']),load_split(paths['val'])
    assert not set(training.subjects).intersection(validation.subjects)
    manifest={'config':asdict(config),'sources':source_hashes(),'versions':package_versions(),
              'training':training.fingerprint,'selection_validation':validation.fingerprint,
              'split':'Original Training vs original Validation; fixed subject-disjoint split; no exclusions.',
              'protocol':'Selection-validation optimization. Validation labels tune hyperparameters, global voting weights, score transforms and decision offsets. Not an independent test.',
              'MMSE_used':False,'LLM_or_paid_API_calls':False}
    manifest_path=output/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('Code/data/config changed. Use a fresh output_dir to avoid mixing experiments.')
    write_json(manifest_path,manifest)
    if (output/'summary.json').exists(): return json.loads((output/'summary.json').read_text())
    write_json(output/'split_subjects.json',{'train':list(training.subjects),'selection':list(validation.subjects)})
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    search=Search(training,validation,config,output)
    search.fit_bank()
    winner=search.select_recipe()
    selected={}
    for key in sorted(recipe_ids(winner['recipe'])):
        spec=Spec(**search.records[key]['spec'])
        print(f'Fitting saved champion component: {spec.task}/{spec.family}/{spec.view}',flush=True)
        with threadpool_limits(limits=config.threads):
            selected[key]=fit_candidate(training,spec,config.threads)
            p=selected[key].predict(validation.representations,validation.subjects)
        if not np.allclose(p,search.predictions[key],rtol=1e-4,atol=1e-6):
            raise RuntimeError('Refitted model did not reproduce cached selection predictions; no champion claim written.')
    champion=Champion(selected,winner['recipe'],np.array(winner['offsets']),
                      {'manifest':manifest,'metric_label':'selection_validation_only'})
    scores,prediction=champion.predict(validation.representations,validation.subjects)
    metrics=measure(validation.y,scores,prediction)
    status=target_status(metrics,config.threshold)
    joblib.dump(champion,output/'champion_selection_only.joblib',compress=3)
    if status['macro_target_met']:
        joblib.dump(champion,output/'target_met_selection_only.joblib',compress=3)
    names=np.array(['CN','MCI','DEM'])
    frame=pd.DataFrame(scores,index=validation.subjects,columns=['score_CN','score_MCI','score_DEM'])
    frame.index.name='subject_hash';frame['true']=names[validation.y];frame['predicted']=names[prediction]
    frame.to_csv(output/'selection_predictions.csv')
    summary={'metric_scope':'SELECTION VALIDATION: reused for model/weight/threshold selection, not unseen-test performance',
             'selection_metrics':metrics,'target_status':status,'successful_unique_candidates':len(search.records),
             'threshold':config.threshold,
             'failed_candidates':len(search.failures),'subjects_train':len(training.subjects),
             'subjects_selection':len(validation.subjects),'subject_overlap':0,'MMSE_used':False,
             'collection_context_features_used':sorted({c for model in selected.values()
                                                       for c in model.processor.columns if c.startswith('context__')}),
             'recipe':winner['recipe'],'offsets':winner['offsets'],
             'saved_model':str(output/'champion_selection_only.joblib')}
    from .report import save_report
    save_report(output,summary,validation.y,scores,prediction,search.records)
    write_json(output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--data-root')
    parser.add_argument('--output-dir',default='results_v3/run_full')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--trials',type=int,default=160,help='Flat and CN/MCI search budgets; DEM uses at least half.')
    parser.add_argument('--recipe-trials',type=int,default=2500)
    parser.add_argument('--ensemble-trials',type=int,default=1200)
    args=parser.parse_args()
    if args.prepare_only:
        from .data import prepare
        prepare(args.data_root)
    else:
        run(Config(output_dir=args.output_dir,threads=args.threads,flat_trials=args.trials,
                   cn_mci_trials=args.trials,dem_trials=max(0,int(args.trials*5/8)),
                   recipe_trials=args.recipe_trials,ensemble_trials=args.ensemble_trials),args.data_root)


if __name__=='__main__':
    main()
