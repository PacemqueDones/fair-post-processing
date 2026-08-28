from fairpp.postprocessor import FairPostProcessor
from fairpp.model import ThresholdRatioModel

from fairpp.objectives.objectives import (
    CrossEntropyObjective,
    LaplacianFairnessObjective
)

from fairpp.selectors.selectors import TopsisSelector

from fairpp.metrics.metrics import (
    BalancedAccuracyMetric,
    DemographicParityMetric,
    EqualityOpportunityMetric
)

from fairpp.laplacian.builder import MahalanobisLaplacianBuilder

from pprep.pipeline import prepare_dataset_from_yaml
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score
)

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# -----------------------------------------------------------------------------#

# -----------------------------------------------------------------------------#
def ddp(y_true, y_pred, sensitive_features):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    s = np.array(sensitive_features)

    group_0 = s == 0
    group_1 = s == 1

    if group_0.sum() == 0 or group_1.sum() == 0:
        return 0.0

    prob_1_g0 = y_pred[group_0].mean()
    prob_1_g1 = y_pred[group_1].mean()

    return float(abs(prob_1_g1 - prob_1_g0))


def deo(y_true, y_pred, sensitive_features):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred)
    s = np.array(sensitive_features)

    group_0 = s == 0
    group_1 = s == 1

    mask_g0 = group_0 & (y_true == 1)
    mask_g1 = group_1 & (y_true == 1)

    if mask_g0.sum() == 0 or mask_g1.sum() == 0:
        return 0.0

    tpr_g0 = y_pred[mask_g0].mean()
    tpr_g1 = y_pred[mask_g1].mean()

    return float(abs(tpr_g1 - tpr_g0))


def calculate_metrics(y_true, y_pred, sensitive_features):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "rec": float(recall_score(y_true, y_pred, zero_division=0)),
        "prec": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ddp": float(ddp(y_true, y_pred, sensitive_features)),
        "deo": float(deo(y_true, y_pred, sensitive_features)),
    }

# -----------------------------------------------------------------------------#

# -----------------------------------------------------------------------------#
datasets = [
    "adult",
    "bank",
    "celeba",
    "compas",
    "dutch",
    "heart_failure"
]

selection_setups = {
    "select_bacc_ddp": {
        "selector": TopsisSelector([1, 1]),
        "metrics": [
            BalancedAccuracyMetric(),
            DemographicParityMetric()
        ]
    },

    "select_bacc_deo": {
        "selector": TopsisSelector([1, 1]),
        "metrics": [
            BalancedAccuracyMetric(),
            EqualityOpportunityMetric()
        ]
    },

    "select_bacc_ddp_deo": {
        "selector": TopsisSelector([1, 1, 1]),
        "metrics": [
            BalancedAccuracyMetric(),
            DemographicParityMetric(),
            EqualityOpportunityMetric()
        ]
    }
}

laplacian_setups = {
    "lap_with_sensitive": {
        "use_sensitive": True
    },
    "lap_without_sensitive": {
        "use_sensitive": False
    }
}
# -----------------------------------------------------------------------------#

# -----------------------------------------------------------------------------#

results = []

id_output_dir = datetime.today().strftime("%Y-%m-%d_%H-%M")
output_dir = Path("experiments/results_fairpp") / id_output_dir
output_dir.mkdir(parents=True, exist_ok=True)

print("Salvando em:", output_dir)

# -----------------------------------------------------------------------------#

# -----------------------------------------------------------------------------#

for dataset in datasets:
    print()
    print("=" * 80)
    print("DATASET:", dataset)
    print("=" * 80)

    for run in range(20):
        print()
        print(f"Run {run + 1}/20")

        data = prepare_dataset_from_yaml(dataset, False)

        sensitive_cols = data["sensitive_cols"]

        X_train = data["X_train"]
        X_val = data["X_val"]
        X_test = data["X_test"]

        y_train = data["y_train"].to_numpy().ravel()
        y_val = data["y_val"].to_numpy().ravel()
        y_test = data["y_test"].to_numpy().ravel()

        s_val = data["s_val"].to_numpy().ravel()
        s_test = data["s_test"].to_numpy().ravel()

        base_model = LogisticRegression(
            max_iter=10000,
            tol=1e-3,
            random_state=run
        )

        base_model.fit(X_train, y_train)

        probs_val = base_model.predict_proba(X_val)
        probs_test = base_model.predict_proba(X_test)
        preds_base = base_model.predict(X_test)

        metrics_base = calculate_metrics(y_test, preds_base, s_test)

        results.append({
            "dataset": dataset,
            "run": run,
            "laplacian_setup": "baseline",
            "uses_sensitive_in_laplacian": "baseline",
            "selection": "baseline",
            "objective": "baseline",
            "solution_type": "baseline",
            "thresholds": None,
            **metrics_base
        })

        for laplacian_name, laplacian_cfg in laplacian_setups.items():

            use_sensitive_in_laplacian = laplacian_cfg["use_sensitive"]

            print()
            print("-" * 80)
            print("Laplaciano:", laplacian_name)
            print("Usa sensível no Laplaciano?", use_sensitive_in_laplacian)
            print("-" * 80)

            if use_sensitive_in_laplacian:
                X_val_lap = X_val.copy()
            else:
                X_val_lap = X_val.drop(
                    columns=sensitive_cols,
                    errors="ignore"
                )

            print("Colunas sensíveis:", sensitive_cols)
            print("X_val shape:", X_val.shape)
            print("X_val_lap shape:", X_val_lap.shape)

            builder = MahalanobisLaplacianBuilder(
                theta=1.0,
                tau_quantile=0.25
            )

            L_val = builder.build(X_val_lap)

            print("L_val shape:", tuple(L_val.shape))
            print("probs_val shape:", probs_val.shape)

            if L_val.shape[0] != probs_val.shape[0]:
                raise ValueError(
                    f"Erro de dimensão em {dataset}: "
                    f"L_val={L_val.shape}, probs_val={probs_val.shape}"
                )

            for selection_name, setup in selection_setups.items():
                print()
                print(f"Seleção: {selection_name}")

                motor = ThresholdRatioModel(num_classes=2, alpha=0.5)

                post = FairPostProcessor(
                    model=motor,
                    objectives=[
                        CrossEntropyObjective(),
                        LaplacianFairnessObjective(
                            L=L_val,
                            fairness_weight=1.0,
                            ce_weight=0.1,
                            normalize=True,
                            symmetrize=True
                        )
                    ],
                    selector=setup["selector"],
                    selection_metrics=setup["metrics"],
                    lr=0.5e-3,
                    epochs=300,
                    track_gradients=False
                )

                post.fit(probs_val, y_val, s_val)

                preds_post = post.predict(probs_test)
                metrics_post = calculate_metrics(y_test, preds_post, s_test)

                results.append({
                    "dataset": dataset,
                    "run": run,
                    "laplacian_setup": laplacian_name,
                    "uses_sensitive_in_laplacian": use_sensitive_in_laplacian,
                    "selection": selection_name,
                    "objective": "laplacian_fairness",
                    "solution_type": "post",
                    "thresholds": str(post.get_thresholds().tolist()),
                    **metrics_post
                })

                print("Thresholds:", post.get_thresholds())
                print("Baseline:", metrics_base)
                print("Post:", metrics_post)

# -----------------------------------------------------------------------------#

# -----------------------------------------------------------------------------#

df = pd.DataFrame(results)

df.to_csv(output_dir / "resultados_detalhados.csv", index=False)

metric_cols = ["acc", "bacc", "rec", "prec", "f1", "ddp", "deo"]

summary = (
    df.groupby(
        [
            "dataset",
            "laplacian_setup",
            "uses_sensitive_in_laplacian",
            "selection",
            "objective",
            "solution_type"
        ],
        as_index=False
    )[metric_cols]
    .agg(["mean", "std"])
)

summary.columns = [
    f"{col[0]}_{col[1]}" if col[1] != "" else col[0]
    for col in summary.columns.to_flat_index()
]

summary = summary.rename(columns={
    "dataset_": "dataset",
    "laplacian_setup_": "laplacian_setup",
    "uses_sensitive_in_laplacian_": "uses_sensitive_in_laplacian",
    "selection_": "selection",
    "objective_": "objective",
    "solution_type_": "solution_type"
})

summary.to_csv(output_dir / "resumo_media_desvio.csv", index=False)

formatted_df = summary[
    [
        "dataset",
        "laplacian_setup",
        "uses_sensitive_in_laplacian",
        "selection",
        "objective",
        "solution_type"
    ]
].copy()

for metric in metric_cols:
    formatted_df[metric] = (
        summary[f"{metric}_mean"].map(lambda x: f"{x:.4f}") +
        " ± " +
        summary[f"{metric}_std"].map(lambda x: f"{x:.4f}")
    )

formatted_df.to_csv(output_dir / "resumo_formatado.csv", index=False)

print()
print("Arquivos salvos em:", output_dir.resolve())
print("- resultados_detalhados.csv")
print("- resumo_media_desvio.csv")
print("- resumo_formatado.csv")