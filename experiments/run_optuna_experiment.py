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
    DEOMetric
    )
from fairpp.diagnose import diagnose_postprocessor

from pprep.pipeline import prepare_dataset_from_yaml

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    accuracy_score,
    recall_score, 
    precision_score, 
    f1_score
    )
import numpy as np

import optuna

#-----------------------------------------------------------------------------
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
    
    recall_g1 = y_pred[(group_1) & (y_true == 1)].mean()
    recall_g0 = y_pred[(group_0) & (y_true == 1)].mean()
    return float(abs(recall_g1 - recall_g0)) 

def calculate_metrics(y_true, y_pred, sensitive_features):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
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
#probs_test = model.predict_proba(X_test)

def objective(trial):
    fairness_weight = trial.suggest_float("fairness_weight", 1.0, 50.0, log=True)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    epochs = trial.suggest_int("epochs", 100, 500)
    alpha = trial.suggest_float("alpha", 0.1, 0.9)
    w_acc = trial.suggest_float("w_acc", 0.0, 1.0)
    w_f1  = trial.suggest_float("w_f1",  0.0, 1.0)
    w_ddp = trial.suggest_float("w_ddp", 0.0, 1.0)
    w_deo = trial.suggest_float("w_deo", 0.0, 1.0)

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
        selector=ZenithSelector([w_acc, w_f1, w_ddp, w_deo]),
        selection_metrics=[
            AccuracyMetric(), F1ScoreMetric(),
            DemographicParityMetric(), DEOMetric()
        ],
        lr=lr,
        epochs=epochs,
        track_gradients=False  # desliga pra ficar mais rápido
    )

    post.fit(probs_val, y_val, s_val)
    metrics = post.best_metrics_

    # Dois objetivos conflitantes → fronteira de Pareto no Optuna também
    accuracy = metrics["f1"]  # ajuste ao nome real da sua métrica
    ddp = metrics["ddp"]
    deo = metrics["deo"]

    return accuracy, ddp, deo

study = optuna.create_study(directions=["maximize", "minimize", 'minimize'])
study.optimize(objective, n_trials=50)

pareto_trials = study.best_trials  # trials não-dominados

print(pareto_trials, type(pareto_trials))

'''print(f"Trials na fronteira de Pareto: {len(pareto_trials)}")
for t in pareto_trials:
    print(t.values)'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

accs = [t.values[0] for t in study.trials if t.values]
ddps = [t.values[1] for t in study.trials if t.values]
deos = [t.values[2] for t in study.trials if t.values]

pareto_accs = [t.values[0] for t in study.best_trials]
pareto_ddps = [t.values[1] for t in study.best_trials]
pareto_deos = [t.values[2] for t in study.best_trials]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(ddps, deos, accs, alpha=0.4, label='Todos trials')
ax.scatter(pareto_ddps, pareto_deos, pareto_accs, color='red', s=80, label='Pareto front')
ax.set_xlabel('DDP (minimizar)')
ax.set_ylabel('DEO (minimizar)')
ax.set_zlabel('Accuracy (maximizar)')
plt.legend()
plt.show()