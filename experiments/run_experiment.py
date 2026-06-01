
from fairpp.model import  (
    ThresholdRatioModel, 
    ThresholdRatioSiLUModel, 
    ThresholdRatioDGateModel
)

from fairpp.objectives.objectives import (
    CrossEntropyObjective,
    DemographicParityObjective,
    EqualityOpportunityObjective,
    DemographicParityKLObjective,
    EqualityOpportunityKLObjective
)

from fairpp.metrics.metrics import (
    BalancedAccuracyMetric,
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    F1ScoreMetric,
    DemographicParityMetric,
    EqualityOpportunityMetric
)

from fairpp.selectors.selectors import (
    TopsisSelector, 
    ZenithSelector
)

from fairpp.diagnose import diagnose_postprocessor

from fairpp.postprocessor import FairPostProcessor

from pprep.pipeline import prepare_dataset_from_yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score
import numpy as np

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
    bacc = balanced_accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    diff_dp = ddp(y_true, y_pred, sensitive_features)
    diff_eo = deo(y_true, y_pred, sensitive_features)

    return {
        "acc": float(acc),
        'bacc': float(bacc),
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

motor = ThresholdRatioDGateModel(num_classes=2, alpha=0.5)

post = FairPostProcessor(
    model=motor,
    objectives=[CrossEntropyObjective(), EqualityOpportunityObjective(fairness_weight = 8.0, ce_weight=0.5)],
    selector=TopsisSelector([1, 1]),
    selection_metrics=[BalancedAccuracyMetric(), EqualityOpportunityMetric()],
    lr=.5e-2,
    epochs=300,
    track_gradients=False
)

post.fit(probs_val, y_val, s_val)
preds = post.predict(probs_test)

print(post.get_thresholds())
print()
print("Thresholds: ", post.get_thresholds())
print()
print("Soloção com post-processing: ", calculate_metrics(y_test, preds, s_test))
print("Soloção sem post-processing: ", calculate_metrics(y_test, model.predict(X_test), s_test))
print()
pareto_front_unico = np.unique(post.pareto_front_, axis=0)
for point in pareto_front_unico:
    print(point)
print()
for point in post.pareto_points_:
    print(point)

import matplotlib.pyplot as plt
pareto_front_unique = np.unique(post.pareto_front_, axis=0)
x = 1 - pareto_front_unique[:, 0]
y = pareto_front_unique[:, 1]

# Criando o gráfico
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='royalblue', linestyle='--', marker='o', label='Fronteira de Pareto')

# Estilização básica
plt.title('Fronteira de Pareto', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Métrica Acurácia)', fontsize=11)
plt.ylabel('Métrica Fairness', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

# Exibir o gráfico
plt.tight_layout()
plt.show()

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