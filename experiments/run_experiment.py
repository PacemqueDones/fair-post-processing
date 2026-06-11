
from fairpp.model import  (
    ThresholdRatioModel, 
    ThresholdRatioSiLUModel, 
    ThresholdRatioDGateModel
)

from fairpp.laplacian.builder import MahalanobisLaplacianBuilder

from fairpp.objectives.objectives import (
    CrossEntropyObjective,
    DemographicParityObjective,
    EqualityOpportunityObjective,
    DemographicParityKLObjective,
    EqualityOpportunityKLObjective,
    LaplacianFairnessObjective
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
from sklearn.model_selection import train_test_split, StratifiedKFold

import torch
import numpy as np
from datetime import datetime
from pathlib import Path

#-----------------------------------------------------------------------------
# Funções auxiliares para avaliação
#-----------------------------------------------------------------------------
def resumir_resultados(lista_resultados):
    keys = lista_resultados[0].keys()
    resumo = {}

    for key in keys:
        valores = np.array([r[key] for r in lista_resultados])

        resumo[key] = {
            "mean": float(valores.mean()),
            "std": float(valores.std(ddof=1))
        }

    return resumo

def calculate_metrics(y_true, y_pred, sensitive_features):
    results = {}

    metrics = [
        AccuracyMetric(),
        BalancedAccuracyMetric(),
        RecallMetric(),
        PrecisionMetric(),
        F1ScoreMetric(),
        DemographicParityMetric(),
        EqualityOpportunityMetric(),
    ]

    # garante tensores
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true)

    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred)

    if not torch.is_tensor(sensitive_features):
        sensitive_features = torch.tensor(sensitive_features)

    # garante vetor 1D
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    sensitive_features = sensitive_features.view(-1)

    for metric in metrics:
        results[metric.name] = float(
            round(
                metric(
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_attr=sensitive_features
                )
            ,4)
        )

    return results

#-----------------------------------------------------------------------------
# Configurações do experimento
#-----------------------------------------------------------------------------

DATASET = "adult"

RANDOM_STATE = 42
TEST_SIZE = 0.20
K = 5

USE_SENSITIVE_IN_MODEL = True
USE_SENSITIVE_IN_LAPLACIAN = False

experiment_date = datetime.now().strftime("%d/%m/%Y-%H-%M")
experiment_dir = Path("experiments") / "results" / "one_datasets" / DATASET / experiment_date

#-----------------------------------------------------------------------------
# Dados completos
#-----------------------------------------------------------------------------

data = prepare_dataset_from_yaml(DATASET, False)

target_col = data["target_col"]
sensitive_cols = data["sensitive_cols"]

X_full = data["X_full_processed"]
y_full = data["y_full"].to_numpy().ravel()
s_full = data["s_full"].to_numpy().ravel()


#-----------------------------------------------------------------------------
# Divisão e preparação do dados
#-----------------------------------------------------------------------------

idx_all = np.arange(len(y_full))

train_idx, test_idx = train_test_split(
    idx_all,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_full
)

X_test = X_full.iloc[test_idx].copy()
y_test = y_full[test_idx]
s_test = s_full[test_idx]

if USE_SENSITIVE_IN_MODEL:
    X_test_model = X_test.copy()
else:
    X_test_model = X_test.drop(columns=sensitive_cols, errors="ignore")


#-----------------------------------------------------------------------------
# K-Fold dentro do treino
#-----------------------------------------------------------------------------

skf = StratifiedKFold(
    n_splits=K,
    shuffle=True,
    random_state=RANDOM_STATE
)

resultados_post = []
resultados_base = []

posts = []
models = []

for fold, (train_pos, val_pos) in enumerate(skf.split(train_idx, y_full[train_idx]), start=1):

    print("=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)

    fold_train_idx = train_idx[train_pos]
    fold_val_idx = train_idx[val_pos]

    X_train = X_full.iloc[fold_train_idx].copy()
    X_val = X_full.iloc[fold_val_idx].copy()

    y_train = y_full[fold_train_idx]
    y_val = y_full[fold_val_idx]

    s_train = s_full[fold_train_idx]
    s_val = s_full[fold_val_idx]

    #-------------------------------------------------------------------------
    # Modelo base: com ou sem atributo sensível
    #-------------------------------------------------------------------------
    if USE_SENSITIVE_IN_MODEL:
        X_train_model = X_train.copy()
        X_val_model = X_val.copy()
    else:
        X_train_model = X_train.drop(columns=sensitive_cols, errors="ignore")
        X_val_model = X_val.drop(columns=sensitive_cols, errors="ignore")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_model, y_train)

    probs_val = model.predict_proba(X_val_model)
    probs_test = model.predict_proba(X_test_model)

    preds_base = model.predict(X_test_model)

    #-------------------------------------------------------------------------
    # Laplaciano: com ou sem atributo sensível
    #-------------------------------------------------------------------------

    if USE_SENSITIVE_IN_LAPLACIAN:
        X_val_lap = X_val.copy()
    else:
        X_val_lap = X_val.drop(columns=sensitive_cols, errors="ignore")

    builder = MahalanobisLaplacianBuilder(
        theta=1.0,
        tau_quantile=0.25
    )

    L_val = builder.build(X_val_lap)

    #-------------------------------------------------------------------------
    # Pós-processador
    #-------------------------------------------------------------------------

    motor = ThresholdRatioDGateModel(num_classes=2, alpha=0.5)

    post = FairPostProcessor(
        model=motor,
        objectives=[
            CrossEntropyObjective(),
            LaplacianFairnessObjective(L=L_val)
        ],
        selector=TopsisSelector([1, 1]),
        selection_metrics=[
            BalancedAccuracyMetric(),
            EqualityOpportunityMetric()
        ],
        lr=.5e-2,
        epochs=300,
        track_gradients=True
    )

    post.fit(probs_val, y_val, s_val)

    fold_dir = experiment_dir / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    print(fold_dir)

    diagnose_postprocessor(
        post=post,
        output_dir=fold_dir
    )

    preds_post = post.predict(probs_test)

    #-------------------------------------------------------------------------
    # Avaliação no teste externo
    #-------------------------------------------------------------------------

    metricas_post = calculate_metrics(y_test, preds_post, s_test)
    metricas_base = calculate_metrics(y_test, preds_base, s_test)

    resultados_post.append(metricas_post)
    resultados_base.append(metricas_base)

    posts.append(post)
    models.append(model)

    print("Thresholds:", post.get_thresholds())
    print()
    print("Solução com post-processing:", metricas_post)
    print("Solução sem post-processing:", metricas_base)
    print()

#-----------------------------------------------------------------------------
# Resumo dos 5 folds no teste externo
#-----------------------------------------------------------------------------

resumo_post = resumir_resultados(resultados_post)
resumo_base = resumir_resultados(resultados_base)

print("=" * 70)
print("RESUMO FINAL - POST-PROCESSING")
print("=" * 70)

for metrica, valores in resumo_post.items():
    print(f"{metrica}: {valores['mean']:.4f} ± {valores['std']:.4f}")

print()

print("=" * 70)
print("RESUMO FINAL - BASELINE")
print("=" * 70)

for metrica, valores in resumo_base.items():
    print(f"{metrica}: {valores['mean']:.4f} ± {valores['std']:.4f}")