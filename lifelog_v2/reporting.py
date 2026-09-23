"""Save honest labels, confusion matrices, ROC curves and executed reports."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, roc_auc_score

from .metrics import NAMES, measure


def evaluation_figure(y, scores, prediction, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y, prediction, labels=[0, 1, 2]), display_labels=NAMES).plot(
        ax=axes[0], colorbar=False, cmap="Blues", values_format="d")
    axes[0].set_title("Subject-level decisions")
    for k, name in enumerate(NAMES):
        fpr, tpr, _ = roc_curve(np.asarray(y) == k, scores[:, k])
        auc = roc_auc_score(np.asarray(y) == k, scores[:, k])
        axes[1].plot(fpr, tpr, label=f"{name}: AUC {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="One-vs-rest ROC")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(output, summary, validation_y, scores, prediction, nested_result):
    output = Path(output)
    evaluation_figure(validation_y, scores, prediction, "Official validation (previously viewed in v1)", output / "validation_evaluation.png")
    if nested_result is not None:
        evaluation_figure(*nested_result, "Nested outer OOF: subject-disjoint evaluation", output / "nested_evaluation.png")
    ablation = pd.read_csv(output / "development" / "ablation.csv")
    fig, ax = plt.subplots(figsize=(11, 10))
    positions = np.arange(len(ablation))
    ax.barh(positions - .18, ablation.raw_roc_auc_macro_ovr, height=.35, label="AUC macro OVR")
    ax.barh(positions + .18, ablation.raw_macro_recall, height=.35, label="Macro recall, before offsets")
    ax.set_yticks(positions, ablation.name)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.axvline(.8, color="black", linestyle="--", linewidth=1)
    ax.set_title("Development ablations: same subjects and folds; selection-biased")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "development_ablation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    rows = [
        ("v1 개발 OOF (선택 편향 있음)", .559757, .486821, .493644, .531915, .611765, .404255, .444444),
        ("v1 공식 Validation", .595454, .427350, .498551, .545455, .615385, 0., .666667),
    ]
    names = {"development_oof_selection_biased": "v2 개발 OOF (선택 편향 있음)",
             "nested_outer_oof": "v2 중첩 CV 바깥 OOF", "official_validation_reused_from_v1": "v2 기존 공식 Validation 재평가"}
    for key, name in names.items():
        metrics = summary[key]
        if metrics is not None:
            rows.append((name, *[metrics[x] for x in ["roc_auc_macro_ovr", "macro_recall", "macro_f1", "accuracy", "recall_CN", "recall_MCI", "recall_DEM"]]))
    table = "| 평가 | ROC-AUC | Macro recall | Macro F1 | Accuracy | CN recall | MCI recall | DEM recall |\n"
    table += "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    table += "\n".join("| " + str(r[0]) + " | " + " | ".join(f"{v:.4f}" for v in r[1:]) + " |" for r in rows)
    text = "# 실행된 v2 결과\n\n" + table + "\n\n"
    text += f"공식 Validation macro 기준 달성: **{summary['validation_target']['macro_target_met']}**. "
    text += f"중첩 CV와 Validation에서 모두 macro 기준 달성: **{summary['both_evaluations_macro_target_met']}**.\n\n"
    text += f"Validation의 세 클래스 recall 모두 0.8 이상: **{summary['validation_target']['all_classes_target_met']}**.\n\n"
    text += "- 기준: macro one-vs-rest ROC-AUC ≥ 0.8 및 macro recall ≥ 0.8. Macro recall은 balanced accuracy와 같습니다.\n"
    text += "- 모든 점수는 피험자 단위입니다. 구간별 점수를 피험자별로 평균한 뒤 평가했습니다.\n"
    text += "- 개발 OOF는 탐색·앙상블·결정 임계값 선택에 재사용되어 낙관적일 수 있습니다. 중첩 CV 바깥 fold는 이 선택에서 제외했습니다.\n"
    text += "- 중첩 CV는 최종 탐색보다 학습 인원과 Optuna 예산이 작으므로 완전히 같은 학습 규모의 추정치는 아닙니다.\n"
    text += "- 공식 Validation은 v1에서 이미 확인했으므로 새 미사용 외부 검증이 아닙니다. v2는 그 결과로 설정을 선택하지 않았습니다.\n"
    text += "- Validation MCI 4명, DEM 3명이며 bootstrap 구간은 표본 조건부 구간입니다. 학습·튜닝 불확실성을 모두 포함하지 않습니다.\n"
    text += "- 클래스 offset은 결정에만 적용합니다. ROC-AUC는 원래 score로 계산하며 score를 임상 위험 확률로 해석하지 않습니다.\n\n"
    text += "## 산출물\n\n- `candidate_model.joblib`: 점수와 관계없이 재현용 후보 모델.\n"
    text += "- `validation_target_met_model.joblib`: Validation macro 기준을 통과했을 때만 저장.\n"
    text += "- `cv_and_validation_target_met_model.joblib`: 두 평가에서 macro 기준을 모두 통과했을 때만 저장.\n"
    text += "- `development/ablation.csv`: 27개 재학습 비교 및 부모 조건 대비 차이. 고정 reference는 v1의 최적 모델 복제품이 아닙니다.\n"
    text += "- `development/search_results.csv`, `development/selected.json`: 전체 탐색과 확정된 설정.\n"
    text += "- `nested/outer_oof_predictions.csv`, `nested/outer_metrics.csv`: 바깥 fold 예측과 성능.\n"
    text += "- `validation_predictions.csv`, `validation_bootstrap_ci.csv`: Validation 예측과 불확실성.\n\n"
    text += "## 선택된 모델\n\n```json\n" + json.dumps({k: summary[k] for k in ["selected_specs", "weights", "offsets"]}, indent=2, ensure_ascii=False) + "\n```\n"
    (output / "RESULTS.md").write_text(text)
