# V2 결과와 V3 실행 안내

V2는 목표에 미달했습니다. V3는 **피험자 분리와 MMSE 제외**를 유지하면서, 고정된 Validation 점수를 직접 최적화하는 탐색으로 변경했습니다. 특징 추출과 코드 검증은 완료했고 **실제 데이터의 V3 분류기 학습은 아직 실행하지 않았습니다.** 따라서 V3가 0.8에 도달했다는 결과는 없습니다.

## V2에서 확인한 병목

출처: `results_v2/run_full/summary.json`과 후보별 탐색 기록.

| 평가 | ROC-AUC macro OVR | Macro recall | Macro F1 |
|---|---:|---:|---:|
| 개발 OOF: 설정 선택에 사용 | 0.604705 | 0.568906 | 0.514875 |
| Nested outer OOF | 0.543056 | 0.441849 | 0.392027 |
| 공식 Validation | 0.613575 | 0.446581 | 0.481746 |

| 공식 Validation 클래스 | 인원 | Recall | OVR AUC |
|---|---:|---:|---:|
| CN | 26 | 0.423077 | 0.593407 |
| MCI | 4 | 0.250000 | 0.336207 |
| DEM | 3 | 0.666667 | 0.911111 |

DEM의 순위 구분력보다 CN–MCI 구분이 문제입니다. MCI는 4명 중 1명만 맞혔습니다. DEM을 잘 구분하는 binary 모델만으로 3-class 목표가 해결되지는 않습니다. V2의 통합 Optuna 탐색은 동일한 LDA 점수에 반복적으로 머문 경우도 있어, V3는 모델 계열마다 예산을 배정합니다.

## 팀 실험·논문에서 반영한 점

- `Google-Ajou-AICapstone/Hyunsoo/final_dementia_screening_model.py`의 활동량 변동성 기반 DEM 판별을 단일 특징 기준선으로 반영했습니다. 해당 코드의 높은 binary 점수에는 특정 DEM 피험자 제외가 포함되어 있어 그대로 3-class 근거로 사용하지 않습니다. V3는 그 피험자도 유지합니다.
- `Google-Ajou-AICapstone/Hyunsoo/reproduction_audit/REPORT.md`와 `code/ymj_faithful_build.py`를 참고해 수면 중 심박 감소, 밤 사이·날짜 사이 변동성을 추가했습니다. 해당 재현 코드를 실행하거나 CognitiveFunction 입력을 가져오지 않고, 활동·수면 파일의 허용 열만 읽는 별도 구현을 만들었습니다.
- 로컬 `papers/`의 Kim·Park 논문은 CN 111명/MCI 51명의 **binary** 연구입니다. 보고된 AUC 0.861, sensitivity 0.820은 3-class 성능이 아닙니다. 논문의 정확한 모든 수식이 공개된 것은 아니므로 V3 특징은 명시적인 재구성입니다. [논문 DOI](https://doi.org/10.3349/ymj.2025.0575).
- 팀의 `SangHyo/3-class/ThreeClass_TransformerTabNet_Google/20260720_005257_full_clinical_plus_lifelog_archive/training/FINAL_REPORT.json`의 0.84848은 MMSE 포함 실험의 accuracy입니다. AUC·macro recall 0.8 달성의 근거로 사용하지 않습니다.
- 로컬 VAE 논문에서 확인한 일별 행 단위 분할 결과도 피험자 분리를 보장한 3-class 결과로 간주하지 않았습니다.

논문과 팀 실험은 유망한 특징·모델 구조를 찾는 데 사용했습니다. 기존 결과 중 조건이나 지표가 다른 수치를 V3 성능으로 옮기지 않았습니다.

## V3의 평가 방식

| 항목 | 구현 |
|---|---|
| Training | 원래 Training 141명: CN 85, MCI 47, DEM 9 |
| 선택용 Validation | 원래 Validation 33명: CN 26, MCI 4, DEM 3 |
| 피험자 분리 | 교집합 0명; 동일인의 날짜·윈도우는 같은 쪽에 유지 |
| 학습 | 대치·특징 선택·스케일·PCA·증강·분류기는 Training에서만 fit |
| 선택 | Validation 정답으로 후보, score 변환, 결합 가중치, 결정 offset 선택 |
| CV | 기본 실행에 K-fold / nested CV 없음 |
| MMSE | 파일·점수·문항 및 그 파생 특징 제외 |
| 목표 | macro OVR AUC ≥0.8 **그리고** macro recall ≥0.8 |
| 추가 표시 | 각 클래스 recall도 모두 ≥0.8인지 별도 표시 |

이 Validation은 의도적으로 **선택용**입니다. 결과 파일에 기록되는 값은 독립 테스트 성능이 아닙니다. 이것은 일반화 추정에 예산을 쓰지 않고 주어진 분할의 성능에 집중하겠다는 요청을 반영한 설정입니다. 높은 점수의 분할을 찾기 위한 피험자 재배치나 오분류 피험자 제외는 하지 않습니다.

Macro recall은 세 클래스 recall의 단순 평균입니다. 예를 들어 이 분할에서 각 클래스 recall이 0.8 이상이려면 CN은 최소 21/26명, MCI는 4/4명, DEM은 3/3명을 맞혀야 합니다. Macro 지표 달성과 세 클래스 개별 달성을 구분해 저장합니다.

## 특징과 ablation

현재 추출된 통합 후보 특징은 **5,144개**입니다. 매번 전부 학습시키는 것이 아니라 Training에서 순위를 매겨 일부를 선택하거나 축소합니다.

| 특징 집합 | 후보 개수 |
|---|---:|
| V2 기존 특징 | 1,333 |
| V3 날짜·심박·수면 변동성 특징 | 3,799 |
| 웨어러블 수집 시점 특징 | 12 |
| 합계 | 5,144 |

새 특징에는 심박 감소율·수면시간당 감소량·최저 심박의 시점, 자정 경계를 고려한 취침 시각, 수면 중간 시각, MAD·CV·연속 날짜 변화량, 7/14/28일 이동 변동성, 추세 및 기간별 변화가 포함됩니다. 결측 날짜를 건너뛰어 연속 날짜 차이를 계산하지 않습니다. 심박 baseline은 원래 첫 15분을 유지하는 정의와 팀의 첫 유효 3개 표본 정의를 별도로 제공합니다.

수집 시작·끝·중앙 날짜, 계절, 주말 비율도 두 제약에 저촉되지 않는 후보로 추가했습니다. 이는 **수집 시점 정보이며 생리 지표가 아닙니다.** 포함/제외/단독 사용을 명시적으로 비교하고, 최종 모델이 이를 선택했는지도 기록합니다. 이메일·피험자 해시·진단 날짜는 모델 입력이 아닙니다.

기본 탐색은 다음과 같습니다.

1. **56개 고정 후보**: 활동·수면·변동성·심박 감소·일주기·소수 특징, 직접 3-class와 두 binary 분기, 14/28일 윈도우 집계.
2. 이 안의 **16개 통제 ablation**: 같은 CatBoost 기준 설정에서 특징 집합, 수집 시점, 클래스 가중치, SMOTE/jitter, 스케일, 특징 선택법, 특징 개수를 각각 바꿉니다. 결과와 기준 대비 차이를 별도 표로 저장합니다.
3. **420회 Optuna**: 직접 3-class 160회, DEM-vs-rest 100회, CN-vs-MCI 160회. 10개 모델 계열에 각각 예산을 배분합니다.
4. 직접 모델과 `P(DEM)`, `P(MCI | CN/MCI)`의 계층 결합을 비교합니다. 계층 score는 `[(1-d)(1-q), (1-d)q, d]`입니다.
5. 최대 2,500개 score 변환과 1,200개 soft-voting 가중치 조합을 추가로 탐색합니다. 최상위 조합은 고정 격자 외에 분류가 바뀌는 score 경계 사이의 결정 offset도 탐색합니다.

모델 계열: CatBoost, XGBoost, LightGBM, ExtraTrees, RandomForest, HistGradientBoosting, LogisticRegression, RBF-SVM, shrinkage LDA, RBF kernel ridge.

특징 선택은 ANOVA·mutual information·Training 내부 RFE를 비교합니다. 증강의 부모는 Training 피험자만 허용하며, 윈도우 행은 피험자별 총 가중치를 맞춥니다. 결정 offset은 recall에 적용하며, AUC는 최종 결합 score에서 계산합니다. 이 score를 보정된 임상 위험 확률이라고 주장하지 않습니다.

## 실행

프로젝트 루트에서 `CN_MCI_DEM_PerformanceSearch_NoMMSE_v3.ipynb`를 열고 `.venv` Python 커널로 실행하세요. **‘실제 학습 시작’ 셀부터 장시간 작업**입니다. 현재 특징 캐시가 있어 앞부분은 재사용합니다. 노트북을 모두 실행하면 학습도 시작됩니다.

터미널에서는 다음 한 줄로 같은 기본 탐색을 실행할 수 있습니다.

```bash
.venv/bin/python -u -m lifelog_v3.search --threads 4
```

이 Python 학습 코드는 LLM/유료 API를 호출하지 않습니다. 학습의 수치 연산 자체는 Codex 토큰을 사용하지 않습니다. Codex가 로그를 읽고 분석하는 대화에는 토큰이 사용되므로, 장시간 실행은 위 명령이나 노트북으로 직접 진행하도록 준비했습니다.

다른 환경에서는 Python 3.12 환경에서 `python -m pip install -r requirements_v3.txt`로 설치하세요. 현재 작업 환경에는 추가 라이브러리와 macOS용 `libomp` 설치 및 import 확인을 마쳤습니다. 다른 macOS에서 OpenMP 오류가 나면 `brew install libomp`가 필요할 수 있습니다. [XGBoost 공식 문서](https://xgboost.readthedocs.io/en/stable/python/python_intro.html), [LightGBM 공식 API](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html).

중단 후 **같은 코드·데이터·설정·출력 폴더**로 재실행하면 완료된 후보의 예측 캐시와 Optuna SQLite를 재사용합니다. 코드·환경 버전·예산을 바꾸면 혼합을 막기 위해 새 출력 폴더가 필요합니다. 예산 확대 예:

```bash
.venv/bin/python -u -m lifelog_v3.search --trials 320 --recipe-trials 5000 --ensemble-trials 2400 --output-dir results_v3/run_expanded
```

확대 실행은 새 탐색입니다. 먼저 기본 실행 결과를 확인하는 것이 좋습니다. 동일 출력 폴더의 동시 실행은 잠금으로 차단합니다. 다른 데이터 위치는 `--data-root` 또는 노트북 `DATA_ROOT`로 지정할 수 있습니다.

## 저장되는 결과

기본 출력 위치는 `results_v3/run_full/`입니다.

| 파일 | 내용 |
|---|---|
| `summary.json`, `RESULTS.md` | 최종 선택용 점수, 클래스별 지표, 목표 달성 여부, 수집 시점 특징 사용 여부 |
| `champion_selection_only.joblib` | 목표 달성 여부와 관계없이 최고 선택점수 모델 |
| `target_met_selection_only.joblib` | 두 macro 지표가 모두 0.8 이상일 때만 추가 저장 |
| `selection_predictions.csv` | 해시화된 피험자별 정답·최종 예측·세 score |
| `controlled_ablations.csv` | 16개 통제 ablation과 기준 대비 차이 |
| `candidate_selection_scores.csv` | 모든 성공한 학습 후보의 선택용 점수 |
| `recipe_selection_scores.csv`, `best_by_search_stage.csv` | 직접·계층·변환·결합·결정 offset 탐색 결과 |
| `classification_report.csv`, `confusion_matrix.csv`, `selection_evaluation.png` | 분류 지표와 ROC/혼동행렬 |
| `manifest.json`, `split_subjects.json` | 코드·환경·데이터 지문, 피험자 분리 기록 |
| `failures.json`, `trials_*.csv`, `studies.sqlite3` | 실패 원인과 재시작 가능한 탐색 기록 |

최종 선택된 구성요소는 Training만으로 다시 fit하고 캐시 예측이 재현되는지 확인한 후 저장합니다. 모델은 `predict_csv(activity_csv, sleep_csv)`로 정답 없이 새 활동·수면 파일을 예측할 수 있습니다. 피험자 해시는 분리 확인·결과 연결에만 사용합니다.

## 완료한 검증과 남은 실행

- 실제 자료: 141/33명, 피험자 교집합 0, 양쪽 특징 열 일치, 모든 계획된 특징 집합 존재 확인.
- 분류기 fit을 금지한 계약 검사: 날짜 결측·심박 위치·전처리·증강 부모·분기·임계값·저장 경로 등 검증.
- 작은 인공 데이터의 native smoke 검사: 10개 모델 계열 × 3개 분기와 RFE 사례, 총 31개 후보의 fit·예측 형식·저장·재로딩 확인. 이 인공 데이터 점수는 성능 결과로 저장하지 않았습니다.
- 노트북은 실제 학습 셀을 제외하고 실행 검증합니다. 기존 V1/V2 결과와 데이터는 변경하지 않습니다.

검증 기록은 `results_v3/preparation.json`, `schema_checks.json`, `contract_checks.json`, `native_smoke.json`, `notebook_checks.json`에 저장합니다. 실제 V3 목표 달성 여부는 사용자가 장시간 학습을 완료한 뒤 생성되는 `run_full/summary.json`에서 확인해야 합니다.
