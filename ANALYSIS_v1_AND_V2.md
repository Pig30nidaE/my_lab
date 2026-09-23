# 기존 실행 결과와 v2 실행 안내

기존 모델은 요청하신 ROC-AUC·recall 0.8 기준에 미달합니다. 새 버전은 **특징 추출과 학습 없는 검증까지 완료**했으며, 사용자 요청에 따라 **모델 학습·성능 ablation은 실행하지 않았습니다. v2 성능은 아직 모릅니다.**

## 1. 기존 노트북 결과

출처: `CN_MCI_DEM_GroupKFold_NoMMSE.ipynb`의 실행 출력, 특히 cell index 12·14·19. 원본 노트북은 수정하지 않았습니다.

| 평가 | ROC-AUC (macro OVR) | Macro recall = balanced accuracy | Macro F1 | Accuracy | CN recall | MCI recall | DEM recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 개발 OOF, 141명 | 0.559757 | 0.486821 | 0.493644 | 0.531915 | 0.611765 | 0.404255 | 0.444444 |
| 공식 Validation, 33명 | 0.595454 | 0.427350 | 0.498551 | 0.545455 | 0.615385 | 0.000000 | 0.666667 |

Recall의 평균 방식은 요청에서 지정되지 않았습니다. CN이 많은 데이터이므로 세 클래스를 동등하게 평가하는 **macro recall**을 주기준으로 사용하고, 클래스별 recall도 함께 보고합니다. 기존 결과는 weighted/micro recall(단일 라벨 분류에서는 accuracy와 같음)이나 개별 클래스 recall로 해석해도 0.8 기준을 만족하지 않습니다.

Validation 혼동행렬은 저장된 피험자별 예측에서 다음과 같습니다. 행은 실제 클래스, 열은 예측 클래스입니다.

| 실제 \ 예측 | CN | MCI | DEM |
|---|---:|---:|---:|
| CN | 16 | 10 | 0 |
| MCI | 4 | 0 | 0 |
| DEM | 0 | 1 | 2 |

MCI 4명은 모두 CN으로 분류되었고, CN 10명은 MCI로 분류되었습니다. DEM은 3명 중 2명을 맞혔지만 이 표본에서 한 명 차이가 recall 33.3%p를 만듭니다. 전원을 CN으로 예측할 때 Validation accuracy는 26/33=0.7879이므로, accuracy만으로 좋은 모델이라고 판단할 수 없습니다.

노트북에 저장된 Validation의 피험자 stratified bootstrap 95% 구간은 다음과 같습니다.

| 지표 | 추정치 | 하한 | 상한 |
|---|---:|---:|---:|
| Macro ROC-AUC | 0.595454 | 0.492202 | 0.705415 |
| Macro recall | 0.427350 | 0.217949 | 0.589744 |
| DEM recall | 0.666667 | 0.000000 | 1.000000 |

이는 클래스별 인원을 고정한 조건부 bootstrap 구간이며 학습 표본 변화나 튜닝 불확실성을 모두 포함하지 않습니다.

### 해석

- 학습 데이터의 실제 독립 표본은 **141명(CN 85, MCI 47, DEM 9)**입니다. 9,705개 일별 기록이 9,705명의 독립 표본을 뜻하지 않습니다. 670개 특징에 비해 사람 수가 적고 DEM이 특히 적습니다.
- Validation은 **33명(CN 26, MCI 4, DEM 3)**이며 split 간 동일 피험자는 없습니다. 코드상 ID는 집계·분할에만 사용하며 MMSE 파일은 읽지 않습니다.
- 개발 OOF와 Validation의 AUC가 모두 낮습니다. 특정 holdout에서만 점수가 떨어진 상황보다, 현재 특징·학습 방법의 분류 신호가 충분하지 않은 상황에 가깝습니다. 원인을 표본 수 하나로 단정할 수는 없습니다.
- 기존 모델은 **macro-F1을 최적화**했습니다. 사용자가 요구한 AUC와 recall의 동시 달성과는 목적함수가 다릅니다.
- 기존 탐색·특징 선택 규칙·앙상블·class offset 선택에 동일한 OOF가 재사용되어 개발 OOF에 선택 편향이 있을 수 있습니다. 낮은 OOF를 독립 일반화 성능이라고 확대 해석할 수 없습니다.
- 기존 GroupKFold의 검증 DEM 수는 **3/1/1/3/1**명이었습니다. v2 준비된 층화 피험자 분할에서는 **2/1/2/2/2**명입니다. 분할 변경이 성능 향상을 보장하는 것은 아닙니다.
- 최종 선택에서 `weight_all=1.0`이므로 physiology 전용 모델의 최종 결합 가중치는 0입니다. 활동·수면 permutation F1 감소는 각각 약 0.168·0.162이지만, 이는 특징을 제거하고 재학습한 ablation 결과가 아닙니다.
- 원본 설명에는 '미실행'이라는 이전 문구가 남아 있지만 실제 실행 카운트와 출력이 있습니다. 본 분석은 저장된 실제 출력을 기준으로 했습니다.

## 2. v2에서 준비한 내용

실행 노트북: **`CN_MCI_DEM_SubjectCV_NoMMSE_v2.ipynb`**. 함께 제공한 `lifelog_v2/` 패키지를 같은 프로젝트에 두어야 합니다.

| 표현 | Training 행 | Validation 행 | 특징 수 | 독립 피험자 수 |
|---|---:|---:|---:|---|
| 기존 개인 요약 | 141 | 33 | 670 | 141 / 33 |
| 확장 개인 요약 | 141 | 33 | 1,333 | 141 / 33 |
| 14일 구간 요약 | 784 | 196 | 1,333 | 141 / 33 |
| 28일 구간 요약 | 449 | 110 | 1,333 | 141 / 33 |

실제로 특징 추출을 수행해 `results_v2/cache/`에 저장했습니다. 원본 수면의 동일 피험자·기상 날짜 중복은 Training 11건, Validation 1건이며 기존의 최장 수면 선택 규칙을 유지했습니다. 모든 피험자가 구간 표현에서도 유지되었고 전체기간 요약으로 대체해야 하는 fallback은 0명입니다.

추가 특징은 일별 평균·표준편차·MAD·변동계수·주말/평일 차이 및 24시간대별 MET 평균·표준편차입니다. 특징 정의는 라벨을 사용하지 않습니다. 상수·결측 필터, 순위 선택, 상관 제거, 대치, 스케일링, PCA는 **매 학습 fold에서만 fit**합니다. 준비 단계 전체 데이터 통계는 감사 목적으로만 기록하며 학습 특징 선택에 사용하지 않습니다.

### 27개 고정 ablation + 추가 탐색

- 원본 특징 → 확장 특징, 활동만/수면만, physiology만, quality/vendor/coupling 제외.
- 전체 특징 → 상위 100/50개, 상관 중복 제거, ANOVA → mutual information 순위.
- 개인 요약 → 14일/28일 구간 요약, 산술 평균 → 기하 평균 예측 결합.
- 클래스 보정 없음/제곱근 역빈도/역빈도 가중치.
- CatBoost, logistic regression, RBF SVM, ExtraTrees, RandomForest, histogram gradient boosting, shrinkage LDA, GaussianNB.
- CN 대 인지저하 → MCI 대 DEM의 계층형 CatBoost/logistic regression.
- PCA와 모델별 하이퍼파라미터, OOF soft-voting 앙상블, 클래스별 결정 offset.

`results_v2/ablation_plan.csv`에 조건과 직접 비교할 부모 조건을 저장했습니다. **이것은 실험 계획이며 성능 결과가 아닙니다.** reference 모델은 같은 조건에서의 제거·추가 효과를 보기 위한 고정 설정이고 v1 Optuna 최적 모델을 그대로 재현한 모델은 아닙니다.

창 단위 모델은 학습 가중치를 `1 / 피험자의 창 수 × 클래스별 피험자 수^(-balance)`로 두어 사람마다 기여도를 맞춥니다. 검증에서는 창 예측을 먼저 피험자별로 결합한 뒤 지표를 계산합니다. 창을 독립 피험자로 취급하거나 사람의 창을 fold 사이에 나누지 않습니다.

### 평가와 목표

사용자가 v2에서는 '피험자 분할·MMSE 제외 두 가지만 유지'하도록 허용했으므로 **중첩 CV**를 추가했습니다.

1. Training 141명 내부에서 바깥 5-fold, 안쪽 3-fold로 특징 선택·모델 탐색·앙상블·decision offset을 분리 평가합니다. 바깥 test 피험자는 내부 선택에 관여하지 않습니다.
2. 전체 Training의 5-fold OOF로 최종 설정을 고르고 전체 Training만 사용하여 학습합니다.
3. 고정한 모델을 기존 공식 Validation에서 평가합니다. **이 Validation은 v1에서 이미 확인한 표본이며, 새 미사용 외부 검증으로 부르지 않습니다.** v2 코드에서 Validation 지표로 모델을 다시 선택하는 단계는 없습니다.

최적화는 `min(macro ROC-AUC, macro recall)`을 중심으로 약한 쪽을 끌어올리도록 설정했습니다. 두 지표의 평균과 macro-F1은 작은 tie-breaker입니다. 결정 offset은 내부/development OOF에서만 정하고 ROC-AUC는 offset 적용 전 score로 계산합니다. SVM을 포함한 score는 별도 임상 확률 보정을 거치지 않았습니다.

개발 OOF는 모델·앙상블·offset 선택에 재사용되므로 **선택 편향이 있는 개발 수치**로 표시합니다. 중첩 CV도 최종 탐색보다 Training 크기와 탐색 예산이 작으므로 완전히 같은 학습 규모의 성능 추정치는 아닙니다.

목표는 macro ROC-AUC와 macro recall이 모두 ≥0.8입니다. 세 클래스 각각의 recall이 모두 ≥0.8인지도 별도 표시합니다. 이 데이터와 유한한 탐색으로 목표 달성이나 전역 최적 모델을 보장할 수는 없습니다.

## 3. 실행 방법

현재 `.venv`에는 필요한 패키지가 있습니다. 같은 환경의 정확한 버전은 `requirements_v2.txt`에 기록했습니다. 다른 컴퓨터에서는 해당 파일로 환경을 구성하고 프로젝트 루트에서 실행합니다.

가장 간단한 방법은 새 노트북을 열어 위에서부터 실행하는 것입니다. **학습은 '학습 실행'이라고 표시한 `run_experiment` 셀에서 시작**합니다. 그 앞의 셀은 특징 캐시 로딩과 점검만 수행합니다.

터미널에서 전체 실험을 실행하려면:

```bash
.venv/bin/python -m lifelog_v2.experiments
```

학습 없이 준비·검증만 다시 실행하려면:

```bash
.venv/bin/python -m lifelog_v2.check_preparation
```

기본 설정은 최종 탐색 80 trial, nested 내부 탐색 각각 20 trial, 최종 후보 3 seeds, CPU 4 threads입니다. 고정 ablation까지 포함하면 단일 시드 단계에 **1,240개 후보×fold 평가**가 있고 선택 후보의 시드 반복·전체 학습이 추가됩니다. 계층형 모델의 각 후보는 내부 분류기 2개를 학습합니다. **실제 학습 시간은 측정하지 않았습니다.** 더 큰 탐색은 노트북의 `n_trials`, `nested_trials`로 설정합니다.

각 후보의 OOF 캐시와 Optuna SQLite 기록을 저장하므로 동일한 코드·설정·데이터로 재실행하면 완료한 후보 평가를 재사용합니다. 다른 코드·데이터·설정과 결과가 섞이지 않도록 manifest가 일치해야 합니다. 설정을 변경할 때는 `output_dir`도 새 경로로 지정하세요. 같은 output_dir에서 여러 실험 프로세스를 동시에 실행하지 마세요.

## 4. 저장 방식과 현재 완료 상태

현재 있는 것은 특징 캐시·특징 사전·준비 검사·ablation 계획입니다. **학습된 모델, v2 AUC/recall, ablation 성능표는 아직 없습니다.**

사용자가 학습을 실행한 뒤 `results_v2/run_full/` 아래에 다음 파일을 생성합니다.

| 파일 | 저장 조건/의미 |
|---|---|
| `RESULTS.md`, `summary.json` | 완료한 실험의 성능·판정·주의점 |
| `development/ablation.csv` | 동일 fold에서 재학습한 27개 조건; 기본 결정과 OOF offset 적용 후 성능을 구분 |
| `development/search_results.csv`, `optuna_trials.csv` | 추가 탐색 전체 기록·실패 trial 포함 |
| `nested/outer_oof_predictions.csv`, `outer_metrics.csv` | 모델 선택에서 분리된 바깥 fold의 예측·성능 |
| `validation_predictions.csv`, `validation_bootstrap_ci.csv` | 고정 모델의 Validation 예측·구간 |
| `candidate_model.joblib` | 성공 여부와 무관한 재현용 후보 |
| `validation_target_met_model.joblib` | Validation의 두 macro 지표가 모두 0.8 이상일 때 |
| `cv_and_validation_target_met_model.joblib` | 중첩 CV와 Validation 모두 두 macro 지표가 0.8 이상일 때 |

`results_v2/preparation_checks.json`에 기록된 검사 결과는 passed, classifier fits 0입니다. fold 경계, 가중치, 전처리 분리, 27개 고정 모델의 API, 추가 50개 탐색 설정 생성·생성자 호환성을 확인했습니다. **분류기의 실제 fit 경로와 학습 후 보고서 생성은 실행하지 않았으므로 해당 경로의 실행 검증은 아직 남아 있습니다.**

새 노트북의 학습 호출 셀 1개를 제외한 코드 셀 7개도 실행해 확인했습니다. 이 검사 중에는 8종 분류기의 `fit` API를 호출하면 즉시 오류가 나도록 차단했고 학습 호출은 없었습니다. 결과는 `results_v2/notebook_pretraining_check.json`에 저장했습니다. 노트북 자체의 실행 출력은 비워 두었습니다.

## 공식 문서

- [scikit-learn 중첩/비중첩 CV 비교](https://sklearn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html): 모델 선택과 평가를 같은 CV 결과로 수행할 때 생기는 선택 편향.
- [scikit-learn StratifiedGroupKFold](https://sklearn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html): 그룹 분리와 클래스 비율 보존. v2는 피험자당 한 행의 라벨 테이블을 먼저 층화 분할한 뒤 모든 구간을 해당 피험자의 fold에 연결합니다.
- [CatBoost 공통 파라미터](https://catboost.ai/docs/en/references/training-parameters/common): CPU boosting, 클래스/표본 가중치 등. v2는 클래스 보정과 피험자 보정을 결합한 sample_weight를 직접 전달합니다.
