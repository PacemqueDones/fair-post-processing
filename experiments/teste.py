# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)

from pprep.pipeline import prepare_dataset_from_yaml

from fairpp.model import (
    ThresholdRatioModel,
    ThresholdRatioSiLUModel,
    ThresholdRatioDGateModel,
)

from fairpp.objectives.objectives import (
    CrossEntropyObjective,
    DemographicParityObjective,
    EqualityOpportunityObjective,
    DemographicParityKLObjective,
    EqualityOpportunityKLObjective,
)

from fairpp.metrics.metrics import (
    BalancedAccuracyMetric,
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    F1ScoreMetric,
    DemographicParityMetric,
    EqualityOpportunityMetric,
)

from fairpp.selectors.selectors import (
    TopsisSelector,
    ZenithSelector,
)

from fairpp.postprocessor import FairPostProcessor
from fairpp.diagnose import diagnose_postprocessor


# =============================================================================
# MÉTRICAS AUXILIARES
# =============================================================================

def benefit_binary(y_true, y_pred):
    """
    Calcula o benefício individual usado no paper:
        b_i = y_pred_i - y_true_i + 1

    Espera y_true e y_pred em {0, 1}.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return y_pred - y_true + 1


def generalized_entropy(benefits, alpha=2.0, eps=1e-12):
    """
    Calcula o índice de Entropia Generalizada.

    benefits: vetor b = (b_1, ..., b_n), com b_i >= 0.
    alpha: parâmetro da entropia generalizada.
           No paper, eles usam frequentemente alpha = 2.

    Retorna:
        E_alpha(b)
    """
    b = np.asarray(benefits, dtype=float)

    if np.any(b < 0):
        raise ValueError("Os benefícios precisam ser não negativos.")

    mu = b.mean()

    if mu <= eps:
        raise ValueError("A média dos benefícios precisa ser positiva.")

    if alpha == 0:
        # caso limite alpha -> 0
        return np.mean(np.log((mu + eps) / (b + eps)))

    if alpha == 1:
        # caso limite alpha -> 1, índice de Theil
        ratio = b / (mu + eps)
        return np.mean(ratio * np.log(ratio + eps))

    ratio = b / (mu + eps)

    return np.mean(ratio ** alpha - 1.0) / (alpha * (alpha - 1.0))

def individual_unfairness_ge(y_true, y_pred, alpha=2.0):
    """
    Calcula a unfairness individual/overall do paper
    usando entropia generalizada.
    """
    b = benefit_binary(y_true, y_pred)
    return generalized_entropy(b, alpha=alpha)

def ddp(y_true, y_pred, sensitive_features):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    s = np.array(sensitive_features)

    group_0 = s == 0
    group_1 = s == 1

    prob_1_g0 = y_pred[group_0].mean()
    prob_1_g1 = y_pred[group_1].mean()

    return float(abs(prob_1_g1 - prob_1_g0))


def deo(y_true, y_pred, sensitive_features):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred)
    s = np.array(sensitive_features)

    group_0 = s == 0
    group_1 = s == 1

    recall_g0 = y_pred[group_0 & (y_true == 1)].mean()
    recall_g1 = y_pred[group_1 & (y_true == 1)].mean()

    return float(abs(recall_g1 - recall_g0))


def calculate_metrics(y_true, y_pred, sensitive_features):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    ge = individual_unfairness_ge(y_true, y_pred)
    diff_dp = ddp(y_true, y_pred, sensitive_features)
    diff_eo = deo(y_true, y_pred, sensitive_features)

    return {
        "acc": float(acc),
        "bacc": float(bacc),
        "rec": float(rec),
        "prec": float(prec),
        "f1": float(f1),
        "ge": float(ge),
        "ddp": float(diff_dp),
        "deo": float(diff_eo),
    }


# =============================================================================
# 1. CARREGAMENTO DOS DADOS
# =============================================================================

data = prepare_dataset_from_yaml("adult")

X_train = data["X_train"]
X_val = data["X_val"]
X_test = data["X_test"]

y_train = data["y_train"].to_numpy().ravel()
y_val = data["y_val"].to_numpy().ravel()
y_test = data["y_test"].to_numpy().ravel()

s_train = data["s_train"].to_numpy().ravel()
s_val = data["s_val"].to_numpy().ravel()
s_test = data["s_test"].to_numpy().ravel()


# =============================================================================
# 2. TREINAMENTO DO MODELO BASE
# =============================================================================

base_model = LogisticRegression(max_iter=1000)
base_model.fit(X_train, y_train)


# =============================================================================
# 3. PROBABILIDADES DO MODELO BASE
# =============================================================================

probs_val = base_model.predict_proba(X_val)
probs_test = base_model.predict_proba(X_test)

baseline_preds_test = base_model.predict(X_test)


# =============================================================================
# 4. CONFIGURAÇÃO DO MODELO DE THRESHOLD
# =============================================================================

threshold_model = ThresholdRatioDGateModel(
    num_classes=2,
    alpha=0.5,
)


# =============================================================================
# 5. CONFIGURAÇÃO DO PÓS-PROCESSADOR
# =============================================================================

post = FairPostProcessor(
    model=threshold_model,
    objectives=[
        CrossEntropyObjective(),
        EqualityOpportunityObjective(
            fairness_weight=8.0,
            ce_weight=2,
        ),
    ],
    selector=TopsisSelector([1, 1]),
    selection_metrics=[
        BalancedAccuracyMetric(),
        EqualityOpportunityMetric(),
    ],
    lr=0.5e-2,
    epochs=300,
    track_gradients=False,
)


# =============================================================================
# 6. TREINAMENTO DO PÓS-PROCESSADOR NO CONJUNTO DE VALIDAÇÃO
# =============================================================================

post.fit(probs_val, y_val, s_val)


# =============================================================================
# 7. PREDIÇÃO NO CONJUNTO DE TESTE
# =============================================================================

post_preds_test = post.predict(probs_test)


# =============================================================================
# 8. AVALIAÇÃO FINAL
# =============================================================================

post_metrics = calculate_metrics(
    y_true=y_test,
    y_pred=post_preds_test,
    sensitive_features=s_test,
)

baseline_metrics = calculate_metrics(
    y_true=y_test,
    y_pred=baseline_preds_test,
    sensitive_features=s_test,
)


# =============================================================================
# 9. RESULTADOS
# =============================================================================

print("Thresholds:", post.get_thresholds())
print()

print("Solução com post-processing:")
print(post_metrics)
print()

print("Solução sem post-processing:")
print(baseline_metrics)
print()

# =============================================================================
# 9. FROMTE DE UNICO DE PARETO
# =============================================================================
print("Pontos ulteriores a Pareto (validação):")
for point in post.pareto_points_:
    print(point)

print("Pontos da Froteira de Pareto (validação):")
for point in post.pareto_front_:
    print(point)

import matplotlib.pyplot as plt
pareto = np.array(post.pareto_points_)  # converte a lista para array NumPy

x = pareto[:, 0]
y = pareto[:, 1]

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='royalblue', linestyle='--', marker='o', label='Fronteira de Pareto')
plt.xlabel('BACC')
plt.ylabel('GE')
plt.title('Fronteira de Pareto')
plt.legend()
plt.show()

from fairpp.pareto import pareto_front as get_pareto_front

# =============================================================================
# BASELINE
# =============================================================================

baseline_preds_val = base_model.predict(X_val)
baseline_preds_test = base_model.predict(X_test)

baseline_val_metrics = calculate_metrics(y_val, baseline_preds_val, s_val)
baseline_test_metrics = calculate_metrics(y_test, baseline_preds_test, s_test)

print("Baseline validação:")
print(baseline_val_metrics)
print()

print("Baseline teste:")
print(baseline_test_metrics)
print()


# =============================================================================
# FRONTEIRA DE PARETO
# =============================================================================

points = np.asarray(post.pareto_points_, dtype=float)
directions = post.metric_directions_
metric_names = post.metric_names_

front_idx = get_pareto_front(points, directions)
front_raw = points[front_idx]

print("Métricas da seleção:", metric_names)
print("Direções:", directions)
print()

print("Tamanho da fronteira antes do filtro:", len(front_raw))
print("Tamanho da fronteira única antes do filtro:", len(np.unique(front_raw, axis=0)))
print()


# =============================================================================
# FILTRO ANTES DO REESCALAMENTO
# =============================================================================
# Regra usada:
# - métricas de minimização não podem ser piores que o baseline;
# - pelo menos uma métrica de minimização precisa melhorar.
#
# No seu caso:
# - bacc: max
# - deo: min
#
# Então o filtro mantém pontos com deo <= deo_baseline
# e exige algum ganho real em deo.

baseline_selection_point = []

for name in metric_names:
    baseline_selection_point.append(baseline_val_metrics[name])

baseline_selection_point = np.asarray(baseline_selection_point, dtype=float)

keep = np.ones(len(front_raw), dtype=bool)
min_cols = []

tol = 1e-12

for j, direction in enumerate(directions):
    if direction == "min":
        min_cols.append(j)
        keep = keep & (front_raw[:, j] <= baseline_selection_point[j] + tol)

if len(min_cols) > 0:
    min_cols = np.asarray(min_cols)
    improve_min = np.any(
        front_raw[:, min_cols] < baseline_selection_point[min_cols] - tol,
        axis=1
    )
    keep = keep & improve_min

front_filtered = front_raw[keep]
front_idx_filtered = front_idx[keep]

print("Tamanho da fronteira depois do filtro:", len(front_filtered))
print("Tamanho da fronteira única depois do filtro:", len(np.unique(front_filtered, axis=0)))
print()

if len(front_filtered) == 0:
    raise ValueError("O filtro removeu todos os pontos.")


# =============================================================================
# REMOVER REPETIDOS
# =============================================================================

front_unique, first_pos = np.unique(
    front_filtered,
    axis=0,
    return_index=True
)

front_global_idx = front_idx_filtered[first_pos]

order = np.argsort(front_unique[:, 0])

front_unique = front_unique[order]
front_global_idx = front_global_idx[order]


# =============================================================================
# REESCALAMENTO DA FRONTEIRA FILTRADA
# =============================================================================
# Depois do reescalamento:
# 0 = pior valor da fronteira filtrada
# 1 = melhor valor da fronteira filtrada
#
# Todas as métricas ficam no sentido: maior é melhor.

mins = front_unique.min(axis=0)
maxs = front_unique.max(axis=0)
ranges = maxs - mins

front_scaled = np.zeros_like(front_unique)

for j, direction in enumerate(directions):
    if ranges[j] < 1e-12:
        front_scaled[:, j] = 1.0

    elif direction == "max":
        front_scaled[:, j] = (front_unique[:, j] - mins[j]) / ranges[j]

    elif direction == "min":
        front_scaled[:, j] = (maxs[j] - front_unique[:, j]) / ranges[j]


# =============================================================================
# SELEÇÃO NA FRONTEIRA REESCALADA
# =============================================================================

ideal = np.ones(front_scaled.shape[1])
anti_ideal = np.zeros(front_scaled.shape[1])

dist_ideal = np.linalg.norm(front_scaled - ideal, axis=1)
dist_anti = np.linalg.norm(front_scaled - anti_ideal, axis=1)

topsis_score = dist_anti / (dist_ideal + dist_anti + 1e-12)
zenith_score = -dist_ideal
nash_score = np.prod(front_scaled + 1e-12, axis=1)

idx_topsis = int(np.argmax(topsis_score))
idx_zenith = int(np.argmax(zenith_score))
idx_nash = int(np.argmax(nash_score))

selecoes = {
    "TOPSIS": idx_topsis,
    "Zenith": idx_zenith,
    "Nash": idx_nash,
}

scores = {
    "TOPSIS": topsis_score,
    "Zenith": zenith_score,
    "Nash": nash_score,
}


# =============================================================================
# RESULTADOS DAS SELEÇÕES
# =============================================================================

for nome, idx in selecoes.items():
    global_idx = front_global_idx[idx]

    post.best_thresholds_ = post.threshold_history_[global_idx].detach().clone()
    preds_test = post.predict(probs_test)

    test_metrics = calculate_metrics(y_test, preds_test, s_test)

    print("=" * 70)
    print(nome)
    print("=" * 70)

    print("Índice na fronteira filtrada:", idx)
    print("Época/global idx:", global_idx)
    print("Score:", scores[nome][idx])
    print("Thresholds:", post.best_thresholds_)
    print()

    print("Ponto na validação:")
    print(dict(zip(metric_names, front_unique[idx])))
    print()

    print("Ponto reescalado:")
    print(dict(zip(metric_names, front_scaled[idx])))
    print()

    print("Métricas no teste:")
    print(test_metrics)
    print()