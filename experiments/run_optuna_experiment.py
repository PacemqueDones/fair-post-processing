from fairpp.postprocessor import FairPostProcessor
from fairpp.model import ThresholdRatioModel
from fairpp.objectives.objectives import (
    CrossEntropyObjective, 
    DemographicParityObjective, 
    EqualityOpportunityObjective, 
    DemographicParityKLObjective, 
    EqualityOpportunityKLObjective 
    )
from fairpp.selectors.selectors import (
    TopsisSelector, 
    ZenithSelector
    )
from fairpp.metrics.metrics import (
    AccuracyMetric, 
    PrecisionMetric, 
    RecallMetric, 
    F1ScoreMetric, 
    DemographicParityMetric, 
    EqualityOpportunityMetric
    )
from fairpp.diagnose import diagnose_postprocessor

from pprep.pipeline import prepare_dataset_from_yaml

from sklearn.linear_model import LogisticRegression

import numpy as np

import optuna

#-----------------------------------------------------------------------------
def calculate_metrics(y_true, y_pred, sensitive_features):
    acc = AccuracyMetric(y_true, y_pred, sensitive_features)
    rec = RecallMetric(y_true, y_pred, sensitive_features)
    prec = PrecisionMetric(y_true, y_pred, sensitive_features)
    f1 = F1ScoreMetric(y_true, y_pred, sensitive_features)
    diff_dp = DemographicParityMetric(y_true, y_pred, sensitive_features)
    diff_eo = EqualityOpportunityMetric(y_true, y_pred, sensitive_features)

    return {
        "acc": float(acc),
        "rec": float(rec),
        "prec": float(prec),
        "f1": float(f1),
        "ddp": float(diff_dp),
        "deo": float(diff_eo),
    }
#-----------------------------------------------------------------------------

data = prepare_dataset_from_yaml("adult")

X_train = data['X_train']
X_test = data['X_test']
X_val = data['X_val']

y_train = data['y_train'].to_numpy().ravel()
y_test = data['y_test'].to_numpy().ravel()
y_val = data['y_val'].to_numpy().ravel()

s_train = data['s_train'].to_numpy().ravel()
s_test = data['s_test'].to_numpy().ravel()
s_val = data['s_val'].to_numpy().ravel()

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

probs_val = model.predict_proba(X_val)
probs_test = model.predict_proba(X_test)

def objective(trial):
    fairness_weight = trial.suggest_float("fairness_weight", 1.0, 50.0, log=True)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)

    motor = ThresholdRatioModel(num_classes=2, alpha=alpha)

    post = FairPostProcessor(
        model=motor,
        objectives=[
            CrossEntropyObjective(),
            DemographicParityKLObjective(
                fairness_weight=fairness_weight,
                ce_weight=0.01
            )
        ],
        selector=ZenithSelector([1, 1, 2, 2]),
        selection_metrics=[
            AccuracyMetric(), F1ScoreMetric(),
            DemographicParityMetric(), EqualityOpportunityMetric()
        ],
        lr=lr,
        epochs=150,
        track_gradients=False  # desliga pra ficar mais rápido
    )

    post.fit(probs_val, y_val, s_val)
    metrics = post.best_metrics_

    # Dois objetivos conflitantes → fronteira de Pareto no Optuna também
    acc = metrics["acc"]
    f1 = metrics["f1"]
    ddp = metrics["ddp"]
    deo = metrics["deo"]

    return acc, f1, ddp, deo

study = optuna.create_study(directions=["maximize", "maximize", "minimize", 'minimize'])
study.optimize(objective, n_trials=200)

pareto_trials = study.best_trials  # trials não-dominados

pareto_front = [t.values for t in pareto_trials]
pareto_params = [t.params for t in pareto_trials]
selector = ZenithSelector([1, 1, 1, 1])
pareto_idx = selector.select(pareto_front, ["max", "max", "min", "min"])
best_params = pareto_params[pareto_idx]

fairness_weight = round(best_params['fairness_weight'], 2)
lr = round(best_params.get('lr'), 5)
epochs = best_params.get('epochs', 150)
alpha = round(best_params.get('alpha'), 4)
w_ddp = round(best_params.get('w_ddp', 2))
w_deo = round(best_params.get('w_deo', 2))

print("Best hyperparameters: ", best_params)

motor = ThresholdRatioModel(num_classes=2, alpha=alpha)

post = FairPostProcessor(
    model=motor,
    objectives=[
        CrossEntropyObjective(),
        DemographicParityObjective(
            fairness_weight=fairness_weight,
            ce_weight=0.01
        )
    ],
    selector=ZenithSelector([1, 1, 2, 2]),
    selection_metrics=[
        AccuracyMetric(), F1ScoreMetric(),
        DemographicParityMetric(), EqualityOpportunityMetric()
    ],
    lr=lr,
    epochs=150,
    track_gradients=True
)

post.fit(probs_val, y_val, s_val)
preds = post.predict(probs_test)


diagnose_postprocessor(
    post=post,
    model=model,
    X_val=X_val,
    y_val=y_val,
    s_val=s_val,
    X_test=X_test,
    y_test=y_test,
    s_test=s_test,
    preds=preds
)