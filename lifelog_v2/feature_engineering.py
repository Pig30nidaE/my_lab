"""MMSE-free deterministic activity/sleep feature extraction, derived from v1.
Only allowlisted wearable columns are read. IDs are used for grouping only.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
CFG = dict(observation_days=None, min_days=7, csv_chunksize=256)
CLASS_NAMES = ['CN', 'MCI', 'DEM']
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}
def resolve_paths(root=None):
    bases = [Path(root)] if root is not None else [Path.cwd(), Path('/content')]
    candidates=[]
    for base in bases:
        candidates.extend([base, base/'01.데이터', base/'128.치매 고위험군 라이프로그'/'01.데이터', base/'Data'])
    for base in candidates:
        for source_dir, label_dir, act_dir, sleep_dir in [
            ('원천데이터','라벨링데이터','1.걸음걸이','2.수면'),
            ('SourceData','LabelingData','1.Gait','2.Sleep')]:
            result={}
            for split, folder, prefix in [('train','1.Training','train'),('val','2.Validation','val')]:
                label_file = 'training_label.csv' if split=='train' else 'val_label.csv'
                result[split] = {
                    'activity':base/folder/source_dir/act_dir/f'{prefix}_activity.csv',
                    'sleep':base/folder/source_dir/sleep_dir/f'{prefix}_sleep.csv',
                    'activity_label':base/folder/label_dir/act_dir/label_file,
                    'sleep_label':base/folder/label_dir/sleep_dir/label_file,
                }
            if all(p.is_file() for split in result.values() for p in split.values()):
                return base.resolve(), result
    raise FileNotFoundError('DATA_ROOT를 데이터 폴더로 지정하세요. 활동/수면/두 라벨 CSV만 검색합니다.')

def subject_key(value):
    text = str(value).strip().lower()
    if not text or text in {'nan','none','<na>'}:
        raise ValueError('비어 있는 피험자 ID가 있습니다.')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def read_labels(paths):
    tables=[]
    for name in ['activity_label','sleep_label']:
        t=pd.read_csv(paths[name], dtype=str, encoding='utf-8-sig',
                      usecols=['SAMPLE_EMAIL','DIAG_NM'])
        t['subject']=t['SAMPLE_EMAIL'].map(subject_key)
        t['label']=t['DIAG_NM'].str.strip().str.upper()
        if t['label'].isna().any() or not t['label'].isin(CLASS_NAMES).all():
            raise ValueError('허용되지 않은 라벨/빈 라벨이 있습니다.')
        tables.append(t[['subject','label']])
    labels=pd.concat(tables, ignore_index=True)
    conflicts=labels.groupby('subject')['label'].nunique().gt(1)
    if conflicts.any():
        raise ValueError(f'진단이 충돌하는 피험자 {conflicts.sum()}명: 시점 정보 확인 전 학습 중단.')
    return labels.drop_duplicates('subject').set_index('subject')['label']

ACT_NUM = [
 'activity_average_met','activity_cal_active','activity_cal_total','activity_daily_movement',
 'activity_high','activity_inactive','activity_inactivity_alerts','activity_low','activity_medium',
 'activity_met_min_high','activity_met_min_inactive','activity_met_min_low','activity_met_min_medium',
 'activity_non_wear','activity_rest','activity_score','activity_score_meet_daily_targets',
 'activity_score_move_every_hour','activity_score_recovery_time','activity_score_stay_active',
 'activity_score_training_frequency','activity_score_training_volume','activity_steps','activity_total']

SLEEP_NUM = [
 'sleep_awake','sleep_breath_average','sleep_deep','sleep_duration','sleep_efficiency',
 'sleep_hr_average','sleep_hr_lowest','sleep_light','sleep_onset_latency','sleep_rem',
 'sleep_restless','sleep_rmssd','sleep_score','sleep_score_alignment','sleep_score_deep',
 'sleep_score_disturbances','sleep_score_efficiency','sleep_score_latency','sleep_score_rem',
 'sleep_score_total','sleep_temperature_delta','sleep_temperature_deviation','sleep_total']

SEQ_COLS = {
 'activity': {'state':'CONVERT(activity_class_5min USING utf8)', 'met':'CONVERT(activity_met_1min USING utf8)'},
 'sleep': {'hr':'CONVERT(sleep_hr_5min USING utf8)', 'state':'CONVERT(sleep_hypnogram_5min USING utf8)',
           'hrv':'CONVERT(sleep_rmssd_5min USING utf8)'}}

TIME_COLS = {'activity':['activity_day_start','activity_day_end'],
             'sleep':['sleep_bedtime_start','sleep_bedtime_end']}

DAILY_META={}

AUDIT=[]

def register(name, modality, family='physiology', description=''):
    DAILY_META[name]={'modality':modality,'family':family,'description':description or name}
    return name

def number(value):
    try:
        x=float(value)
        return x if np.isfinite(x) else np.nan
    except (ValueError,TypeError):
        return np.nan

def divide(a,b):
    return a/b if np.isfinite(a) and np.isfinite(b) and b>0 else np.nan

def parse_sequence(value, positive_only=False):
    if value is None or str(value).strip() in {'','nan','...','None'}:
        return np.array([],dtype=float)
    tokens=str(value).strip().split('/')
    # 끝부분 slash의 빈 token만 제거하고 내부 빈 token은 NaN으로 위치를 보존한다.
    while tokens and not tokens[-1].strip():
        tokens.pop()
    arr=np.array([number(x) for x in tokens],dtype=float)
    if positive_only:
        arr[arr<=0]=np.nan
    return arr

def safe_corr(a,b,min_n=4):
    a,b=np.asarray(a,float),np.asarray(b,float)
    good=np.isfinite(a)&np.isfinite(b)
    if good.sum()<min_n or np.std(a[good])==0 or np.std(b[good])==0:
        return np.nan
    return float(np.corrcoef(a[good],b[good])[0,1])

def continuous_stats(a):
    good=a[np.isfinite(a)]
    out={k:np.nan for k in ['mean','std','p10','p90','rmssd_adjacent','lag1','last_first_third']}
    if len(good):
        out.update(mean=float(np.mean(good)),std=float(np.std(good)),
                   p10=float(np.quantile(good,.1)),p90=float(np.quantile(good,.9)))
    if len(a)>1:
        d=np.diff(a); d=d[np.isfinite(d)]
        out['rmssd_adjacent']=float(np.sqrt(np.mean(d*d))) if len(d) else np.nan
        out['lag1']=safe_corr(a[:-1],a[1:])
    third=len(a)//3
    if third and np.isfinite(a[:third]).any() and np.isfinite(a[-third:]).any():
        out['last_first_third']=float(np.nanmean(a[-third:])-np.nanmean(a[:third]))
    return out

def categorical_stats(a,states):
    valid=np.isfinite(a)&np.isin(a,states)
    out={}
    probabilities=[]
    for state in states:
        p=float(np.sum(valid & (a==state))/valid.sum()) if valid.any() else np.nan
        out[f'code_{state}_fraction']=p
        probabilities.append(p)
        lengths=[]; run=0
        for x in a:
            if np.isfinite(x) and x==state:
                run+=1
            elif run:
                lengths.append(run); run=0
        if run: lengths.append(run)
        out[f'code_{state}_longest_minutes']=max(lengths,default=0)*5 if valid.any() else np.nan
    probabilities=np.array(probabilities)
    positive=probabilities[np.isfinite(probabilities)&(probabilities>0)]
    out['entropy']=float(-np.sum(positive*np.log(positive))) if len(positive) else np.nan
    pair=valid[:-1]&valid[1:]
    out['transition_rate']=float(np.mean(a[:-1][pair]!=a[1:][pair])) if pair.any() else np.nan
    return out

def local_timestamp(value):
    t=pd.to_datetime(value,errors='coerce')
    if pd.isna(t): return pd.NaT
    return t.tz_localize('Asia/Seoul') if t.tzinfo is None else t.tz_convert('Asia/Seoul')

def clock_hours(t):
    return t.hour+t.minute/60+t.second/3600 if pd.notna(t) else np.nan

def met_clock_features(a,start_hour):
    # 시퀀스 길이를 24시간으로 강제 보간하지 않고 실제 1분 위치를 사용한다.
    out={}; profile=np.full(24,np.nan)
    hours=(start_hour+np.arange(len(a))/60)%24
    valid=np.isfinite(a)&(a>=0)
    if np.isfinite(start_hour) and valid.sum()>=720 and len(np.unique(np.floor(hours[valid])))>=18:
        for h in range(24):
            mask=valid & (np.floor(hours)==h)
            if mask.sum()>=30: profile[h]=np.mean(a[mask])
        t=2*np.pi*hours[valid]/24
        design=np.column_stack([np.ones(len(t)),np.cos(t),np.sin(t)])
        beta=np.linalg.lstsq(design,a[valid],rcond=None)[0]
        amplitude=float(np.hypot(beta[1],beta[2]))
        out.update(cosinor_amplitude=amplitude,
                   cosinor_relative_amplitude=divide(amplitude,abs(beta[0])),
                   cosinor_phase_cos=divide(beta[1],amplitude),
                   cosinor_phase_sin=divide(beta[2],amplitude))
        day=valid&(hours>=8)&(hours<20)
        night=valid&((hours<8)|(hours>=20))
        out['day_night_difference']=float(np.mean(a[day])-np.mean(a[night])) if day.any() and night.any() else np.nan
    else:
        out.update({k:np.nan for k in ['cosinor_amplitude','cosinor_relative_amplitude',
                    'cosinor_phase_cos','cosinor_phase_sin','day_night_difference']})
    windows={}
    for width,name in [(5,'low5'),(10,'high10')]:
        rolled=np.r_[profile,profile[:width-1]]
        values=[np.mean(rolled[i:i+width]) for i in range(24)
                if np.isfinite(rolled[i:i+width]).all()]
        windows[name]=(min(values) if width==5 else max(values)) if values else np.nan
    out['relative_high10_low5']=divide(windows['high10']-windows['low5'],
                                                windows['high10']+windows['low5'])
    out['low5_met']=windows['low5']; out['high10_met']=windows['high10']
    return out,profile

def extract_day(row,modality):
    out={'subject':subject_key(row['EMAIL'])}
    start=local_timestamp(row[TIME_COLS[modality][0]])
    end=local_timestamp(row[TIME_COLS[modality][1]])
    day=start if modality=='activity' else end
    out['_date']=day.normalize().tz_localize(None) if pd.notna(day) else pd.NaT
    out['_start']=start
    numeric=ACT_NUM if modality=='activity' else SLEEP_NUM
    for col in numeric:
        value=number(row.get(col))
        if 'temperature' not in col and np.isfinite(value) and value<0: value=np.nan
        if col in {'sleep_hr_average','sleep_hr_lowest','sleep_rmssd','sleep_breath_average'} and value==0:
            value=np.nan
        family='vendor' if 'score' in col else ('quality' if 'non_wear' in col else 'physiology')
        name=modality+'__'+col.removeprefix(modality+'_')
        out[register(name,modality,family,col)]=value
    for key,column in SEQ_COLS[modality].items():
        seq=parse_sequence(row.get(column),positive_only=key in {'hr','hrv'})
        if key=='met': seq[seq<0]=np.nan
        if key=='state':
            states=range(6) if modality=='activity' else range(1,5)
            seq[~np.isin(seq,list(states))]=np.nan
            stats=categorical_stats(seq,list(states))
        else:
            stats=continuous_stats(seq)
        for stat,value in stats.items():
            name=f'{modality}__{key}_{stat}'
            out[register(name,modality,description=f'{column}: {stat}')]=value
        for stat,value in [('valid_fraction',np.isfinite(seq).mean() if len(seq) else 0),('length',len(seq))]:
            name=f'{modality}__{key}_{stat}'
            out[register(name,modality,'quality',f'{column}: {stat}')]=value
        if modality=='activity' and key=='met':
            met,profile=met_clock_features(seq,clock_hours(start))
            for stat,value in met.items():
                out[register('activity__met_'+stat,modality,description='MET clock: '+stat)]=value
            out.update({f'_hour_{h:02d}':v for h,v in enumerate(profile)})
    if modality=='activity':
        nonwear=out['activity__non_wear']
        interval_minutes=(end-start).total_seconds()/60 if pd.notna(end) and pd.notna(start) else np.nan
        # activity_day_end는 마지막 1초 직전이므로 약 1440분인 경우만 24시간으로 간주한다.
        full_day=np.isfinite(interval_minutes) and abs(interval_minutes-1440)<1
        wear=1440-nonwear if full_day and np.isfinite(nonwear) and 0<=nonwear<=1440 else np.nan
        out[register('activity__interval_minutes',modality,'quality')]=interval_minutes
        derived={
            'steps_per_wear_hour':divide(out['activity__steps'],wear/60),
            'moderate_high_fraction':divide(out['activity__medium']+out['activity__high'],wear),
            'inactive_fraction':divide(out['activity__inactive'],wear),
        }
    else:
        total=out['sleep__total']; duration=out['sleep__duration']
        derived={name+'_fraction':divide(out['sleep__'+name],total) for name in ['deep','light','rem']}
        derived.update(awake_fraction=divide(out['sleep__awake'],duration),
                       latency_fraction=divide(out['sleep__onset_latency'],duration))
        # 자정 경계의 불연속을 피하기 위한 원형 시각 특징.
        for name,timestamp in [('bedtime',start),('wake',end)]:
            phase=2*np.pi*clock_hours(timestamp)/24
            derived[name+'_sin']=np.sin(phase); derived[name+'_cos']=np.cos(phase)
    for name,value in derived.items():
        out[register(modality+'__'+name,modality,description=name)]=value
    return out

def load_daily(path,modality,split):
    allowed=['EMAIL']+(ACT_NUM if modality=='activity' else SLEEP_NUM)+TIME_COLS[modality]+list(SEQ_COLS[modality].values())
    header=pd.read_csv(path,nrows=0,encoding='utf-8-sig').columns
    missing=set(allowed)-set(header)
    if missing: raise ValueError(f'{modality} 필수 열 누락: {sorted(missing)}')
    records=[]; input_rows=0
    for chunk in pd.read_csv(path,dtype=str,encoding='utf-8-sig',usecols=allowed,chunksize=CFG['csv_chunksize']):
        input_rows+=len(chunk)
        records.extend(extract_day(row,modality) for row in chunk.to_dict('records'))
    frame=pd.DataFrame.from_records(records)
    bad_dates=int(frame['_date'].isna().sum())
    frame=frame.dropna(subset=['_date'])
    if modality=='sleep':
        frame=frame.sort_values(['subject','_date','sleep__duration','_start'],
                    ascending=[True,True,False,True],na_position='last',kind='stable')
    else:
        frame=frame.sort_values(['subject','_date','_start'],kind='stable')
    duplicate=int(frame.duplicated(['subject','_date']).sum())
    frame=frame.drop_duplicates(['subject','_date'],keep='first').drop(columns=['_start'])
    AUDIT.append(dict(split=split,modality=modality,input_rows=input_rows,
                      bad_dates=bad_dates,duplicate_subject_dates=duplicate,kept_rows=len(frame),
                      subjects=frame['subject'].nunique()))
    assert not frame.duplicated(['subject','_date']).any()
    return frame.sort_values(['subject','_date']).reset_index(drop=True)

FEATURE_META_RECORDS={}

AGG_STATS=['median','iqr','p10','p90']

TEMPORAL_KEYS = {
 'activity':['activity__steps','activity__average_met','activity__inactive','activity__met_cosinor_amplitude',
             'activity__met_relative_high10_low5'],
 'sleep':['sleep__total','sleep__efficiency','sleep__hr_average','sleep__rmssd',
          'sleep__awake_fraction','sleep__deep_fraction','sleep__rem_fraction']}

def add_subject_feature(out,name,value,modality,family='physiology',description=''):
    out[name]=float(value) if np.isfinite(value) else np.nan
    FEATURE_META_RECORDS[name]={'modality':modality,'family':family,'description':description or name}

def observation_window(activity,sleep):
    both=pd.concat([activity[['subject','_date']],sleep[['subject','_date']]]).drop_duplicates()
    start=both.groupby('subject')['_date'].min()
    result=[]
    for frame in [activity,sleep]:
        elapsed=(frame['_date']-frame['subject'].map(start)).dt.days
        mask=elapsed.ge(0)
        if CFG['observation_days'] is not None:
            mask &= elapsed.lt(CFG['observation_days'])
        result.append(frame.loc[mask].copy())
    days=pd.concat([x[['subject','_date']] for x in result]).drop_duplicates().groupby('subject').size()
    eligible=days.index[days>=CFG['min_days']]
    return [x[x['subject'].isin(eligible)].copy() for x in result], days

def summarize_modality(frame,subject,modality,out):
    g=frame[frame['subject']==subject].sort_values('_date')
    for col,meta in DAILY_META.items():
        if meta['modality']!=modality: continue
        a=g[col].to_numpy(float) if col in g else np.array([],float)
        finite=a[np.isfinite(a)]
        vals=[np.median(finite),np.quantile(finite,.75)-np.quantile(finite,.25),
              np.quantile(finite,.1),np.quantile(finite,.9)] if len(finite) else [np.nan]*4
        for stat,value in zip(AGG_STATS,vals):
            add_subject_feature(out,col+'__'+stat,value,modality,meta['family'],meta['description']+'; across days '+stat)
        add_subject_feature(out,col+'__missing_fraction',1-len(finite)/len(a) if len(a) else 1,
                            modality,'quality',meta['description']+'; daily missing fraction')
    span=(g['_date'].max()-g['_date'].min()).days+1 if len(g) else 0
    for stat,value in [('n_days',len(g)),('span_days',span),('coverage',divide(len(g),span))]:
        add_subject_feature(out,f'{modality}__quality_{stat}',value,modality,'quality',stat)
    for col in TEMPORAL_KEYS[modality]:
        values=g[col].to_numpy(float) if col in g else np.array([],float)
        times=(g['_date']-g['_date'].min()).dt.days.to_numpy(float) if len(g) else np.array([],float)
        good=np.isfinite(values)
        adjacent=(np.diff(times)==1)&np.isfinite(np.diff(values))
        delta=np.diff(values)[adjacent]
        rmssd=np.sqrt(np.mean(delta**2)) if len(delta) else np.nan
        slope=np.polyfit(times[good],values[good],1)[0] if good.sum()>=4 and np.ptp(times[good])>0 else np.nan
        lag=safe_corr(values[:-1][adjacent],values[1:][adjacent]) if len(values)>1 else np.nan
        for stat,value in [('day_rmssd',rmssd),('trend_per_day',slope),('consecutive_lag1',lag)]:
            add_subject_feature(out,col+'__'+stat,value,modality,description=col+'; '+stat)
    # 일별 시간대 profile의 전체 분산 중 시간대 평균으로 설명되는 비율.
    if modality=='activity':
        hourly=g[[f'_hour_{h:02d}' for h in range(24)]].to_numpy(float) if len(g) else np.empty((0,24))
        valid=np.isfinite(hourly)
        if valid.sum()>48 and np.nanvar(hourly)>0:
            sums=np.nansum(hourly,axis=0); counts=valid.sum(axis=0)
            profile=np.divide(sums,counts,out=np.full(24,np.nan),where=counts>0)
            overall=np.nanmean(hourly)
            between=np.nansum(counts*(profile-overall)**2)
            total=np.nansum((hourly-overall)**2)
            stability=divide(between,total)
        else: stability=np.nan
        add_subject_feature(out,'activity__hourly_profile_stability',stability,modality,
                            description='분산 가중 시간대 반복성; 표준 actigraphy IS와 동일한 지표라고 주장하지 않음')
    if modality=='sleep':
        for phase in ['bedtime','wake']:
            a=g[f'sleep__{phase}_sin'].to_numpy(float) if len(g) else np.array([])
            b=g[f'sleep__{phase}_cos'].to_numpy(float) if len(g) else np.array([])
            mask=np.isfinite(a)&np.isfinite(b)
            strength=np.hypot(a[mask].mean(),b[mask].mean()) if mask.any() else np.nan
            add_subject_feature(out,f'sleep__{phase}_regularity',strength,modality,
                                description='시각 원형 resultant length: 1에 가까울수록 규칙적')

def summarize_coupling(activity,sleep,subject,out):
    a=activity[activity['subject']==subject]
    s=sleep[sleep['subject']==subject]
    # activity date d ↔ wake date d / wake date d+1. 같은 개인 안에서만 결합.
    pairs=[('activity__steps','sleep__total'),('activity__steps','sleep__efficiency'),
           ('activity__average_met','sleep__rmssd'),('activity__inactive','sleep__awake_fraction')]
    for shift,title in [(0,'same_wake_day'),(1,'next_wake_day')]:
        aa=a.copy(); aa['_date']=aa['_date']+pd.Timedelta(days=shift)
        joined=aa.merge(s,on=['subject','_date'],how='inner',validate='one_to_one')
        add_subject_feature(out,f'coupling__{title}_n_pairs',len(joined),'coupling','quality',title+' paired days')
        for ac,sc in pairs:
            corr=safe_corr(joined[ac],joined[sc],min_n=7) if len(joined) else np.nan
            name=f'coupling__{title}__{ac.removeprefix("activity__")}__{sc.removeprefix("sleep__")}_corr'
            add_subject_feature(out,name,corr,'coupling',description=f'{ac} vs {sc}; {title}; Pearson r, min 7 paired days')

def build_subjects(activity,sleep):
    (activity,sleep),all_days=observation_window(activity,sleep)
    subjects=sorted(set(activity['subject'])|set(sleep['subject']))
    rows=[]
    for subject in subjects:
        out={'subject':subject}
        summarize_modality(activity,subject,'activity',out)
        summarize_modality(sleep,subject,'sleep',out)
        summarize_coupling(activity,sleep,subject,out)
        rows.append(out)
    if not rows: raise ValueError('최소 관측일 기준을 만족하는 피험자가 없습니다.')
    features=pd.DataFrame(rows).set_index('subject').sort_index()
    features=features.replace([np.inf,-np.inf],np.nan).astype('float32')
    assert features.index.is_unique and features.columns.is_unique
    return features, activity, sleep, all_days
