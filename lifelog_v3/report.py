"""Reports distinguish selected validation scores from independent test scores."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay,roc_curve,roc_auc_score


def save_report(output,summary,y,p,prediction,records):
    output=Path(output)
    names=['CN','MCI','DEM']
    pd.DataFrame(classification_report(y,prediction,target_names=names,output_dict=True,zero_division=0)).T.to_csv(output/'classification_report.csv')
    matrix=confusion_matrix(y,prediction,labels=[0,1,2])
    pd.DataFrame(matrix,index=names,columns=names).to_csv(output/'confusion_matrix.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    ConfusionMatrixDisplay(matrix,display_labels=names).plot(ax=axes[0],colorbar=False,cmap='Blues')
    for k,name in enumerate(names):
        fpr,tpr,_=roc_curve(np.asarray(y)==k,p[:,k])
        axes[1].plot(fpr,tpr,label=f'{name}: {roc_auc_score(np.asarray(y)==k,p[:,k]):.3f}')
    axes[1].plot([0,1],[0,1],'--',color='gray')
    axes[1].set(xlabel='False positive rate',ylabel='True positive rate',title='One-vs-rest ROC')
    axes[1].legend()
    fig.suptitle('V3 selection validation: used to choose models, weights and decisions')
    fig.tight_layout();fig.savefig(output/'selection_evaluation.png',dpi=160);plt.close(fig)
    recipes=pd.read_csv(output/'recipe_selection_scores.csv')
    stage=recipes.sort_values('objective',ascending=False).groupby('stage',sort=False).first()
    stage.to_csv(output/'best_by_search_stage.csv')
    from .search import controlled_ablation_specs,digest
    comparisons=[]
    reference=records.get(digest(controlled_ablation_specs()['reference'].dictionary())[:24])
    for name,spec in controlled_ablation_specs().items():
        key=digest(spec.dictionary())[:24]
        record=records.get(key)
        row={'ablation':name,'candidate_id':key,'status':'ok' if record else 'failed_or_not_run'}
        if record:
            row.update(record['selection_metrics'])
            if reference:
                for metric in ['roc_auc_macro_ovr','macro_recall']:
                    row['delta_'+metric]=record['selection_metrics'][metric]-reference['selection_metrics'][metric]
        comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(output/'controlled_ablations.csv',index=False)
    feature_rows=[]
    for key,model in records.items():
        for feature in model['selected_features']:
            feature_rows.append({'candidate_id':key,'task':model['spec']['task'],'feature':feature})
    pd.DataFrame(feature_rows).to_csv(output/'candidate_selected_features.csv',index=False)
    m=summary['selection_metrics'];status=summary['target_status']
    text=f'''# V3 실행 결과: 선택용 Validation

**이 점수는 하이퍼파라미터·모델 결합·score 변환·결정 임계값 선택에 사용한 Validation의 성능입니다. 독립 테스트/새 피험자 일반화 성능으로 해석하지 않습니다.**

| 모델 | ROC-AUC macro OVR | Macro recall | 평가 용도 |
|---|---:|---:|---|
| V1 | 0.595454 | 0.427350 | 당시 모델을 고정한 뒤 공식 Validation 평가 |
| V2 | 0.613575 | 0.446581 | 당시 모델을 고정한 뒤 기존 Validation 재평가 |
| V3 | {m['roc_auc_macro_ovr']:.6f} | {m['macro_recall']:.6f} | 이 Validation에서 최고 설정 선택 |

목표 두 macro 지표 ≥{summary['threshold']}: **{status['macro_target_met']}**.
세 클래스 recall 각각 ≥{summary['threshold']} 및 macro AUC ≥{summary['threshold']}: **{status['all_classes_target_met']}**.

| 클래스 | Recall | One-vs-rest AUC | 인원 |
|---|---:|---:|---:|
| CN | {m['recall_CN']:.4f} | {m['auc_CN_ovr']:.4f} | {m['n_CN']} |
| MCI | {m['recall_MCI']:.4f} | {m['auc_MCI_ovr']:.4f} | {m['n_MCI']} |
| DEM | {m['recall_DEM']:.4f} | {m['auc_DEM_ovr']:.4f} | {m['n_DEM']} |

Training {summary['subjects_train']}명, Selection Validation {summary['subjects_selection']}명, 교집합 0명. MMSE 파일·점수·문항은 사용하지 않았습니다. 어려운 피험자를 제외하거나 높은 점수의 split을 골라 쓰지 않았습니다.

최종 모델이 선택한 수집 시기 특징: `{summary['collection_context_features_used']}`. 이 특징은 생리 신호가 아니라 웨어러블 관측의 달력 시점입니다. `controlled_ablations.csv`에서 포함/제외/단독 사용을 비교할 수 있습니다.

성공한 고유 학습 후보 {summary['successful_unique_candidates']}개; 기록된 실패 {summary['failed_candidates']}건. 실패는 `failures.json`과 `trials_*.csv`를 확인하세요.

`champion_selection_only.joblib`은 최고 선택점수 모델입니다. 기준을 달성했을 때만 `target_met_selection_only.joblib`도 저장합니다. `selection_predictions.csv`는 재현 가능한 피험자별 score와 최종 분류입니다. 마지막 전체 Training 재학습 후 캐시 예측 재현을 확인했습니다.

결정 offset은 recall에만 적용됩니다. AUC는 최종 결합·정규화한 score에서 계산합니다. score를 임상적으로 보정된 위험 확률이라고 주장하지 않습니다. 많은 후보와 적은 MCI/DEM Validation 인원 때문에 선택 점수의 낙관성이 큽니다.

## 선택된 결합

```json
{json.dumps({'recipe':summary['recipe'],'offsets':summary['offsets']},ensure_ascii=False,indent=2)}
```
'''
    (output/'RESULTS.md').write_text(text)
