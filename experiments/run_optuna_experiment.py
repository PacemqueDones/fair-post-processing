from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from fairpp.evaluation import calculate_metrics
from fairpp.metrics import (
    AccuracyMetric,
    BalancedAccuracyMetric,
    DemographicParityMetric,
    EqualityOpportunityMetric,
    F1ScoreMetric,
    PrecisionMetric,
    RecallMetric,
)
from fairpp.models import ThresholdCategoricalAdditiveRatioModel
from fairpp.objectives.group import DemographicParityObjective
from fairpp.objectives.performance import CrossEntropyObjective
from fairpp.postprocessor import FairPostProcessor
from fairpp.selection.selectors import TopsisSelector
from pprep.pipeline import prepare_dataset_from_yaml


DATASET = "adult"
RANDOM_STATE = 42
TEST_SIZE = 0.20
POSTPROC_SIZE = 0.35

N_TRIALS = 120
EPOCHS = 800

USE_SENSITIVE_IN_MODEL = True

OUTPUT_DIR = (
    Path("experiments")
    / "results"
    / "one_datasets"
    / DATASET
    / datetime.now().strftime("%d-%m-%Y_%H-%M")
)


metrics = [
    AccuracyMetric(),
    BalancedAccuracyMetric(),
    RecallMetric(),
    PrecisionMetric(),
    F1ScoreMetric(),
    DemographicParityMetric(
        group_reduction="max",
        attribute_reduction="max",
    ),
    EqualityOpportunityMetric(
        group_reduction="max",
        attribute_reduction="max",
    ),
]

selection_metrics = [
    BalancedAccuracyMetric(),
    DemographicParityMetric(
        group_reduction="max",
        attribute_reduction="max",
    ),
]


def infer_category_sizes(sensitive_attr):
    sensitive_attr = np.asarray(sensitive_attr)

    if sensitive_attr.ndim == 1:
        sensitive_attr = sensitive_attr.reshape(-1, 1)

    if sensitive_attr.ndim != 2:
        raise ValueError(
            "sensitive_attr deve possuir shape "
            "(num_samples, num_sensitive_attributes)."
        )

    return [
        np.unique(sensitive_attr[:, j]).size
        for j in range(sensitive_attr.shape[1])
    ]


def build_postprocessor(trial_params, category_sizes):
    motor = ThresholdCategoricalAdditiveRatioModel(
        num_classes=2,
        category_sizes=category_sizes,
        alpha=trial_params["alpha"],
    )

    return FairPostProcessor(
        model=motor,
        objectives=[
            CrossEntropyObjective(),
            DemographicParityObjective(
                fairness_weight=trial_params["fairness_weight"],
                group_reduction="max",
                attribute_reduction="max",
            ),
        ],
        selector=TopsisSelector([1.0, trial_params["selection_ddp_weight"]]),
        selection_metrics=selection_metrics,
        aggregator="upgrad",
        lr=trial_params["lr"],
        epochs=EPOCHS,
    )


def suggest_params(trial):
    return {
        "fairness_weight": trial.suggest_float(
            "fairness_weight",
            1.0,
            100.0,
            log=True,
        ),
        "lr": trial.suggest_float(
            "lr",
            1e-3,
            2e-1,
            log=True,
        ),
        "alpha": trial.suggest_float(
            "alpha",
            0.25,
            8.0,
            log=True,
        ),
        "selection_ddp_weight": trial.suggest_float(
            "selection_ddp_weight",
            1.0,
            8.0,
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = prepare_dataset_from_yaml(DATASET, False)

    target_col = data["target_col"]
    sensitive_cols = data["sensitive_cols"]

    X_full = data["X_full_processed"]
    y_full = data["y_full"].to_numpy().ravel()
    s_full = data["s_full"].to_numpy()

    idx_all = np.arange(len(y_full))

    train_post_idx, test_idx = train_test_split(
        idx_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_full,
    )

    train_idx, post_idx = train_test_split(
        train_post_idx,
        test_size=POSTPROC_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_full[train_post_idx],
    )

    X_train = X_full.iloc[train_idx].copy()
    X_post = X_full.iloc[post_idx].copy()
    X_test = X_full.iloc[test_idx].copy()

    y_train = y_full[train_idx]
    y_post = y_full[post_idx]
    y_test = y_full[test_idx]

    s_post = s_full[post_idx]
    s_test = s_full[test_idx]

    if USE_SENSITIVE_IN_MODEL:
        X_train_model = X_train.copy()
        X_post_model = X_post.copy()
        X_test_model = X_test.copy()
    else:
        X_train_model = X_train.drop(columns=sensitive_cols, errors="ignore")
        X_post_model = X_post.drop(columns=sensitive_cols, errors="ignore")
        X_test_model = X_test.drop(columns=sensitive_cols, errors="ignore")

    base_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    base_model.fit(X_train_model, y_train)

    probs_post = base_model.predict_proba(X_post_model)
    probs_test = base_model.predict_proba(X_test_model)
    preds_base = base_model.predict(X_test_model)

    category_sizes = infer_category_sizes(s_full)

    def objective(trial):
        trial_params = suggest_params(trial)
        post = build_postprocessor(trial_params, category_sizes)
        post.fit(probs_post, y_post, s_post)

        bacc = post.best_metrics_["bacc"]
        ddp = post.best_metrics_["ddp"]

        trial.set_user_attr("selected_epoch", post.history_[post.best_index_]["epoch"])
        trial.set_user_attr("val_bacc", bacc)
        trial.set_user_attr("val_ddp", ddp)
        trial.set_user_attr("val_losses", post.best_losses_)

        return bacc, ddp

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
        multivariate=True,
    )

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=sampler,
    )
    study.optimize(objective, n_trials=N_TRIALS)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(OUTPUT_DIR / "optuna_trials.csv", index=False)

    pareto_trials = study.best_trials
    pareto_points = [trial.values for trial in pareto_trials]
    pareto_params = [trial.params for trial in pareto_trials]

    selector = TopsisSelector([1.0, 4.0])
    selected_idx = selector.select(pareto_points, ["max", "min"])
    best_params = pareto_params[selected_idx]

    final_post = build_postprocessor(best_params, category_sizes)
    final_post.fit(probs_post, y_post, s_post)

    preds_post = final_post.predict(probs_test, s_test)

    metrics_base = calculate_metrics(
        y_true=y_test,
        y_pred=preds_base,
        sensitive_features=s_test,
        metrics=metrics,
    )
    metrics_post = calculate_metrics(
        y_true=y_test,
        y_pred=preds_post,
        sensitive_features=s_test,
        metrics=metrics,
    )

    summary = {
        "dataset": DATASET,
        "target_col": target_col,
        "sensitive_cols": sensitive_cols,
        "random_state": RANDOM_STATE,
        "n_trials": N_TRIALS,
        "epochs": EPOCHS,
        "best_params": best_params,
        "best_validation_metrics": final_post.best_metrics_,
        "base_test_metrics": metrics_base,
        "post_test_metrics": metrics_post,
    }

    pd.Series(summary).to_json(
        OUTPUT_DIR / "selected_solution.json",
        indent=4,
        force_ascii=False,
    )

    print()
    print("=" * 70)
    print("OPTUNA FINALIZADO")
    print("=" * 70)
    print("Diretório:", OUTPUT_DIR)
    print("Melhores hiperparâmetros:", best_params)
    print()
    print("Baseline teste:", metrics_base)
    print("Post teste:", metrics_post)


if __name__ == "__main__":
    main()
