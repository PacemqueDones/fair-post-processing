from fairpp.models import (
    AffineModel,
    CategoricalAdditiveModel,
    CovariateAffineModel,
)

from fairpp.geometry import (
    FairDistanceBuilder,
    FairLaplacianBuilder,
    FairGeometryBuilder,
)

from fairpp.objectives import (
    CrossEntropyObjective,
    KLPreservationObjective,
    LaplacianFairnessObjective
)


from fairpp.metrics import (
    BalancedAccuracyMetric,
    DemographicParityMetric,
    EqualityOpportunityMetric,
    PrecisionMetric,
    RecallMetric,
    IndividualFairnessViolationRateMetric
)

from fairpp.selection.selectors import (
    TopsisSelector,
    ZenithSelector
)

from fairpp.diagnostics import diagnose_postprocessor
from fairpp.evaluation import calculate_metrics, resumir_resultados
from fairpp.postprocessor import FairPostProcessor


import numpy as np
from datetime import datetime
from pathlib import Path

#-----------------------------------------------------------------------------
# Funções auxiliares para avaliação
#-----------------------------------------------------------------------------


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

#-----------------------------------------------------------------------------
# Configurações do experimento
#-----------------------------------------------------------------------------

DATASETS = ["dutch"]
SEEDS = [41]


EPOCHS = 200
LR = 1e-1
ALPHA = 1


PROJECT_DIR = Path(__file__).resolve().parents[2] / 'fair-post-process_experiments' 
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"


#-----------------------------------------------------------------------------
# K-Fold dentro do treino
#-----------------------------------------------------------------------------
resultados_post = []
resultados_base = []

posts = []

for dataset in DATASETS:
    result_date = datetime.now().strftime("%d-%m-%Y")
    result_dir = Path('experiments') / "results" / "individual" / dataset / result_date

    for seed in SEEDS:
        base_dir = ARTIFACTS_DIR / "base" / dataset / f"seed_{seed}"

        base_fold_files = sorted(base_dir.glob("fold_*.npz"))

        if not base_fold_files:
            print(f"Nenhum fold encontrado em {base_dir}")
            continue

        dense_geometry_dir = ARTIFACTS_DIR / "geometry" / dataset / f"seed_{seed}" / "dense"

        dense_geometry_fold_files = sorted(dense_geometry_dir.glob("fold_*.npz"))

        if not dense_geometry_fold_files:
            print(f"Nenhum fold encontrado em {dense_geometry_dir}")
            continue

        for dense_geometry_file, base_file in zip(dense_geometry_fold_files, base_fold_files):
            fold = int(base_file.stem.split("_")[-1])

            print(f"{dataset} | seed {seed} | fold {fold}")

            base_artifact = np.load(base_file)

            X_train=base_artifact["X_train_fair"]
            X_validation=base_artifact["X_validation_fair"]
            X_test=base_artifact["X_test_fair"]

            probs_train = base_artifact["probs_train_oof"]
            probs_validation = base_artifact["probs_validation"]
            probs_test = base_artifact["probs_test"]

            preds_validation_base = base_artifact["preds_validation_base"]
            preds_test_base = base_artifact["preds_test_base"]

            y_train = base_artifact["y_train"]
            y_validation = base_artifact["y_validation"]
            y_test = base_artifact["y_test"]

            S_train = base_artifact["S_train"]
            S_validation = base_artifact["S_validation"]
            S_test = base_artifact["S_test"]

            dense_geometry_artefact = np.load(dense_geometry_file)

            train_L=dense_geometry_artefact["train_L"]
            train_W_vector=dense_geometry_artefact["train_W_vector"]
            train_tau=dense_geometry_artefact["train_tau"]
            validation_N_vector=dense_geometry_artefact["validation_N_vector"]
            test_N_vector=dense_geometry_artefact["test_N_vector"]

            category_sizes = infer_category_sizes(S_train)

            motor = CovariateAffineModel(
                alpha = ALPHA,
                num_classes = 2,
                num_features = X_train.shape[1]
                )

            postprocessor = FairPostProcessor(
                model=motor,
                objectives=[
                    KLPreservationObjective(),
                    LaplacianFairnessObjective(
                        L=train_L,
                        W_vector=train_W_vector,
                        normalize="edges",
                    ),
                ],
                selector=TopsisSelector([1, 1]),
                selection_metrics=[
                    BalancedAccuracyMetric(),
                    IndividualFairnessViolationRateMetric(
                        N_vector=validation_N_vector
                    ),
                ],
                aggregator="upgrad",
                lr=LR,
                epochs=EPOCHS,
            )

            postprocessor.fit(
                train_inputs=probs_train,
                train_y_true=y_train,
                train_sensitive_attr=S_train,
                train_X=X_train,

                val_inputs=probs_validation,
                val_y_true=y_validation,
                val_sensitive_attr=S_validation,
                val_X=X_validation,
            )

            fold_dir = result_dir / f"fold_{fold:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            print(fold_dir)

            diagnose_postprocessor(
                post=postprocessor,
                output_dir=fold_dir
            )

            preds_post = postprocessor.predict(inputs=probs_test, sensitive_attr=S_test, X=X_test)

            #-------------------------------------------------------------------------
            # Avaliação no teste externo
            #-------------------------------------------------------------------------
            metrics = [
                BalancedAccuracyMetric(),
                RecallMetric(),
                PrecisionMetric(),
                DemographicParityMetric(
                    group_reduction="max",
                    attribute_reduction="max",
                    class_reduction="max",
                ),
                EqualityOpportunityMetric(
                    group_reduction="max",
                    attribute_reduction="max",
                    class_reduction="max",
                ),
                IndividualFairnessViolationRateMetric(
                    N_vector=test_N_vector,
                )
            ]

            metricas_post = calculate_metrics(
                y_true=y_test,
                y_pred=preds_post,
                sensitive_features=S_test,
                logits=postprocessor._predict_logits(inputs=probs_test, sensitive_attr=S_test, X=X_test),
                metrics=metrics
            )

            metricas_base = calculate_metrics(
                y_true=y_test,
                y_pred=preds_test_base,
                sensitive_features=S_test,
                logits=np.log(probs_test),
                metrics=metrics
            )

            resultados_post.append(metricas_post)
            resultados_base.append(metricas_base)

            posts.append(postprocessor)

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