from fairpp.postprocessor import FairPostProcessor
from fairpp.model import  ThresholdRatioModel, ThresholdRatioSiLUModel, ThresholdRatioDGateModel
from fairpp.objectives.objectives import CrossEntropyObjective, DemographicParityObjective, EqualityOpportunityObjective, DemographicParityKLObjective, EqualityOpportunityKLObjective
from fairpp.selectors.selectors import TopsisSelector, ZenithSelector
from fairpp.metrics.metrics import AccuracyMetric, PrecisionMetric, RecallMetric, F1ScoreMetric, DemographicParityMetric, EqualityOpportunityMetric
from fairpp.diagnose import diagnose_postprocessor

from pprep.pipeline import prepare_dataset_from_yaml
from sklearn.linear_model import LogisticRegression
from fairpp.engine.nsga2_engine import NSGA2Engine

import torch

import numpy as np

#-----------------------------------------------------------------------------
def calculate_metrics(y_true, y_pred, sensitive_features):
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred)
    sensitive_features = torch.as_tensor(sensitive_features)

    acc = AccuracyMetric()(y_true, y_pred, sensitive_features)
    rec = RecallMetric()(y_true, y_pred, sensitive_features)
    prec = PrecisionMetric()(y_true, y_pred, sensitive_features)
    f1 = F1ScoreMetric()(y_true, y_pred, sensitive_features)
    diff_dp = DemographicParityMetric()(y_true, y_pred, sensitive_features)
    diff_eo = EqualityOpportunityMetric()(y_true, y_pred, sensitive_features)

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

motor = ThresholdRatioDGateModel(num_classes=2, alpha=0.5)

post = FairPostProcessor(
    model=motor,
    objectives=[CrossEntropyObjective(), DemographicParityObjective(fairness_weight = 4.0, ce_weight=0.01)],
    selector=ZenithSelector([1, 1, 2, 2]),
    selection_metrics=[AccuracyMetric(), F1ScoreMetric(), DemographicParityMetric(), EqualityOpportunityMetric()],
    engine=NSGA2Engine(
        population_size=50,
        generations=100,
        seed=42,
    ),
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
for point in post.pareto_points_:
    print(point)

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