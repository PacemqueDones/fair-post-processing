from fairpp.postprocessor import FairPostProcessor
from fairpp.model import ThresholdRatioModel
from fairpp.objectives.objectives import (
    CrossEntropyObjective,
    DemographicParityObjective,
    EqualityOpportunityObjective,
    DemographicParityKLObjective,
    EqualityOpportunityKLObjective
)
from fairpp.selectors.selectors import TopsisSelector, ZenithSelector
from fairpp.metrics.metrics import (
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    F1ScoreMetric,
    DemographicParityMetric,
    DEOMetric
)

from pprep.pipeline import prepare_dataset_from_yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------------
def ddp(y_true, y_pred, sensitive_features):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    s = np.array(sensitive_features)

    group_0 = (s == 0)
    group_1 = (s == 1)

    prob_1_g1 = y_pred[group_1].mean()
    prob_1_g0 = y_pred[group_0].mean()
    return float(abs(prob_1_g1 - prob_1_g0))

def deo(y_true, y_pred, sensitive_features):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred)
    s = np.array(sensitive_features)

    group_0 = (s == 0)
    group_1 = (s == 1)

    # cuidado se algum grupo não tiver exemplos positivos
    mask_g1 = (group_1) & (y_true == 1)
    mask_g0 = (group_0) & (y_true == 1)

    recall_g1 = y_pred[mask_g1].mean() if mask_g1.sum() > 0 else 0.0
    recall_g0 = y_pred[mask_g0].mean() if mask_g0.sum() > 0 else 0.0

    return float(abs(recall_g1 - recall_g0))

def calculate_metrics(y_true, y_pred, sensitive_features):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    diff_dp = ddp(y_true, y_pred, sensitive_features)
    diff_eo = deo(y_true, y_pred, sensitive_features)

    return {
        "acc": float(acc),
        "rec": float(rec),
        "prec": float(prec),
        "f1": float(f1),
        "ddp": float(diff_dp),
        "deo": float(diff_eo),
    }

# -----------------------------------------------------------------------------
def get_selector_name(selector):
    return selector.__class__.__name__.replace("Selector", "").lower()

def get_objective_name(objective):
    """
    Gera nomes amigáveis para as combinações de objetivos.
    """
    if isinstance(objective, list):
        names = []
        for obj in objective:
            if isinstance(obj, DemographicParityObjective):
                names.append("ddp")
            elif isinstance(obj, EqualityOpportunityObjective):
                names.append("deo")
            elif isinstance(obj, DemographicParityKLObjective):
                names.append("ddp_kl")
            elif isinstance(obj, EqualityOpportunityKLObjective):
                names.append("deo_kl")
            else:
                names.append(obj.__class__.__name__.lower())
        return " + ".join(names)

    if isinstance(objective, DemographicParityObjective):
        return "ddp"
    elif isinstance(objective, EqualityOpportunityObjective):
        return "deo"
    elif isinstance(objective, DemographicParityKLObjective):
        return "ddp_kl"
    elif isinstance(objective, EqualityOpportunityKLObjective):
        return "deo_kl"
    else:
        return objective.__class__.__name__.lower()

# -----------------------------------------------------------------------------
datasets = ["adult", "bank", "celeba", "compas", "dutch", "heart_failure"]

objectives = [
    DemographicParityObjective(fairness_weight = 8.0, ce_weight=0.01),
    EqualityOpportunityObjective(fairness_weight = 8.0, ce_weight=0.01),
    DemographicParityKLObjective(fairness_weight = 8.0, ce_weight=0.01),
    EqualityOpportunityKLObjective(fairness_weight = 8.0, ce_weight=0.01),
    [DemographicParityObjective(fairness_weight = 8.0, ce_weight=0.01), EqualityOpportunityObjective(fairness_weight = 8.0, ce_weight=0.01)],
    [DemographicParityKLObjective(fairness_weight = 8.0, ce_weight=0.01), EqualityOpportunityKLObjective(fairness_weight = 8.0, ce_weight=0.01)],
]

selectors = [TopsisSelector([1, 1, 2, 2]), ZenithSelector([1, 1, 2, 2] )]

results = []

output_dir = Path("results_fairpp")
output_dir.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
for dataset in datasets:
    for selector in selectors:
        for objective in objectives:
            for run in range(20):

                data = prepare_dataset_from_yaml(dataset)

                X_train = data['X_train']
                X_test = data['X_test']
                X_val = data['X_val']

                y_train = data['y_train'].to_numpy().ravel()
                y_test = data['y_test'].to_numpy().ravel()
                y_val = data['y_val'].to_numpy().ravel()

                s_train = data['s_train'].to_numpy().ravel()
                s_test = data['s_test'].to_numpy().ravel()
                s_val = data['s_val'].to_numpy().ravel()

                # modelo base
                base_model = LogisticRegression(max_iter=1000)
                base_model.fit(X_train, y_train)

                probs_val = base_model.predict_proba(X_val)
                probs_test = base_model.predict_proba(X_test)

                motor = ThresholdRatioModel(num_classes=2, alpha=.5)

                current_objectives = (
                    [CrossEntropyObjective()] + objective
                    if isinstance(objective, list)
                    else [CrossEntropyObjective(), objective]
                )

                post = FairPostProcessor(
                    model=motor,
                    objectives=current_objectives,
                    selector=selector,
                    selection_metrics=[
                        AccuracyMetric(),
                        F1ScoreMetric(),
                        DemographicParityMetric(),
                        DEOMetric()
                    ],
                    lr=.5e-2,
                    epochs=300,
                    track_gradients=False
                )

                post.fit(probs_val, y_val, s_val)
                preds_post = post.predict(probs_test)
                preds_base = base_model.predict(X_test)

                selector_name = get_selector_name(selector)
                objective_name = get_objective_name(objective)

                metrics_post = calculate_metrics(y_test, preds_post, s_test)
                metrics_base = calculate_metrics(y_test, preds_base, s_test)

                # registro post-processing
                results.append({
                    "dataset": dataset,
                    "selector": selector_name,
                    "objective": objective_name,
                    "run": run,
                    "solution_type": "post",
                    "thresholds": str(post.get_thresholds().tolist()),
                    **metrics_post
                })

                # registro baseline
                results.append({
                    "dataset": dataset,
                    "selector": selector_name,
                    "objective": objective_name,
                    "run": run,
                    "solution_type": "baseline",
                    "thresholds": None,
                    **metrics_base
                })

                print()
                print(f"Dataset: {dataset} | Selector: {selector_name} | Objective: {objective_name} | Run: {run+1}/20")
                print("Thresholds:", post.get_thresholds())
                print("Solução com post-processing:", metrics_post)
                print("Solução sem post-processing:", metrics_base)

# -----------------------------------------------------------------------------
# Tabela detalhada
df = pd.DataFrame(results)
df.to_csv(output_dir / "resultados_detalhados.csv", index=False)

# -----------------------------------------------------------------------------
# Métricas numéricas
metric_cols = ["acc", "rec", "prec", "f1", "ddp", "deo"]

# Média e desvio-padrão
summary = (
    df.groupby(["dataset", "selector", "objective", "solution_type"], as_index=False)[metric_cols]
      .agg(["mean", "std"])
)

# Achata as colunas do MultiIndex
summary.columns = [
    f"{col[0]}_{col[1]}" if col[1] != "" else col[0]
    for col in summary.columns.to_flat_index()
]

# Renomeia as colunas principais para nomes limpos
summary = summary.rename(columns={
    "dataset_": "dataset",
    "selector_": "selector",
    "objective_": "objective",
    "solution_type_": "solution_type"
})

summary.to_csv(output_dir / "resumo_media_desvio.csv", index=False)

# -----------------------------------------------------------------------------
# Tabela formatada com média ± desvio
formatted_df = summary[["dataset", "selector", "objective", "solution_type"]].copy()

for metric in metric_cols:
    formatted_df[metric] = (
        summary[f"{metric}_mean"].map(lambda x: f"{x:.4f}") +
        " ± " +
        summary[f"{metric}_std"].map(lambda x: f"{x:.4f}")
    )

formatted_df.to_csv(output_dir / "resumo_formatado.csv", index=False)

# -----------------------------------------------------------------------------
# Coluna com nome completo do experimento
formatted_df["experiment"] = formatted_df["selector"] + " + " + formatted_df["objective"]

formatted_df[
    ["dataset", "experiment", "solution_type", "acc", "rec", "prec", "f1", "ddp", "deo"]
].to_csv(output_dir / "resumo_experimentos.csv", index=False)

# -----------------------------------------------------------------------------
# Pivot mais seguro
pivot_df = formatted_df.pivot(
    index=["dataset", "selector", "objective"],
    columns="solution_type",
    values=metric_cols
)

# Opcional: achatar colunas do pivot também
pivot_df.columns = [f"{metric}_{sol}" for metric, sol in pivot_df.columns]
pivot_df = pivot_df.reset_index()

pivot_df.to_csv(output_dir / "resumo_pivotado.csv", index=False)

print("\nArquivos salvos em:", output_dir.resolve())
print("- resultados_detalhados.csv")
print("- resumo_media_desvio.csv")
print("- resumo_formatado.csv")
print("- resumo_experimentos.csv")
print("- resumo_pivotado.csv")