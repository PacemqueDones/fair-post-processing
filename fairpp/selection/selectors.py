from typing import Literal

import numpy as np


def _get_directions(metrics):
    """Aceita objetos Metric ou, por compatibilidade, uma lista de direções."""
    if len(metrics) > 0 and isinstance(metrics[0], str):
        directions = list(metrics)
    else:
        directions = [metric.direction for metric in metrics]

    if any(direction not in {"min", "max"} for direction in directions):
        raise ValueError("Cada direção deve ser 'min' ou 'max'.")

    return directions


class TopsisSelector:
    def __init__(self, weights=None):
        self.weights = weights

    def select(self, points, metrics):
        X = np.asarray(points, dtype=float)
        directions = _get_directions(metrics)

        if X.ndim != 2:
            raise ValueError("points deve ter shape (n_points, n_metrics).")

        if len(directions) != X.shape[1]:
            raise ValueError(
                "O número de métricas deve ser igual ao número de colunas."
            )

        weights = (
            np.ones(X.shape[1])
            if self.weights is None
            else np.asarray(self.weights, dtype=float)
        )

        if len(weights) != X.shape[1]:
            raise ValueError(
                "Número de pesos deve ser igual ao número de objetivos."
            )

        # Normalização vetorial por coluna.
        norm = np.linalg.norm(X, axis=0)
        norm[norm == 0] = 1.0
        Xn = X / norm

        # Aplicação dos pesos.
        V = Xn * weights

        ideal_best = []
        ideal_worst = []

        for j, direction in enumerate(directions):
            column = V[:, j]

            if direction == "max":
                ideal_best.append(column.max())
                ideal_worst.append(column.min())
            else:
                ideal_best.append(column.min())
                ideal_worst.append(column.max())

        ideal_best = np.asarray(ideal_best)
        ideal_worst = np.asarray(ideal_worst)

        distance_best = np.linalg.norm(V - ideal_best, axis=1)
        distance_worst = np.linalg.norm(V - ideal_worst, axis=1)

        closeness = distance_worst / (
            distance_best + distance_worst + 1e-12
        )

        return int(np.argmax(closeness))


class ZenithSelector:
    def __init__(self, weights=None):
        self.weights = weights

    def select(self, points, metrics):
        X = np.asarray(points, dtype=float)
        directions = _get_directions(metrics)

        if X.ndim != 2:
            raise ValueError("points deve ter shape (n_points, n_metrics).")

        if len(directions) != X.shape[1]:
            raise ValueError(
                "O número de métricas deve ser igual ao número de colunas."
            )

        weights = (
            np.ones(X.shape[1])
            if self.weights is None
            else np.asarray(self.weights, dtype=float)
        )

        if len(weights) != X.shape[1]:
            raise ValueError(
                "Número de pesos deve ser igual ao número de objetivos."
            )

        # Normalização Min-Max.
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        denominator = maxs - mins
        denominator[denominator == 0] = 1.0
        Xn = (X - mins) / denominator

        zenith = np.asarray([
            1.0 if direction == "max" else 0.0
            for direction in directions
        ])

        distances = np.linalg.norm((Xn - zenith) * weights, axis=1)

        return int(np.argmin(distances))


class ReferenceSelector:
    """
    match:
        Seleciona o ponto mais próximo dos alvos pelo lado aceitável.

    optimize:
        Usa os alvos como restrições e seleciona o ponto mais próximo
        do ideal formado pelas métricas que ficaram livres.
    """

    def __init__(
        self,
        targets: dict[str, float],
        mode: Literal["match", "optimize"] = "match",
        weights: dict[str, float] | None = None,
        p: float = 2.0,
    ):
        if not isinstance(targets, dict):
            raise TypeError("targets deve ser um dicionário.")

        if mode not in {"match", "optimize"}:
            raise ValueError("mode deve ser 'match' ou 'optimize'.")

        if mode == "match" and not targets:
            raise ValueError(
                "O modo 'match' precisa receber pelo menos um alvo."
            )

        if weights is not None and not isinstance(weights, dict):
            raise TypeError("weights deve ser um dicionário ou None.")

        self.targets = {
            name: float(value)
            for name, value in targets.items()
        }
        self.weights = None if weights is None else {
            name: float(value)
            for name, value in weights.items()
        }
        self.mode = mode
        self.p = float(p)

        if any(not np.isfinite(value) for value in self.targets.values()):
            raise ValueError("Todos os alvos devem ser finitos.")

        if self.weights is not None:
            if any(
                not np.isfinite(value) or value <= 0
                for value in self.weights.values()
            ):
                raise ValueError("Todos os pesos devem ser positivos.")

        if self.p != np.inf and self.p < 1:
            raise ValueError("p deve ser maior ou igual a 1, ou np.inf.")

    def _get_weights(self, metric_names):
        if self.weights is None:
            return np.ones(len(metric_names))

        return np.asarray([
            self.weights.get(name, 1.0)
            for name in metric_names
        ])

    def _distance(self, values):
        return np.linalg.norm(values, ord=self.p, axis=1)

    def select(self, points, metrics):
        X = np.asarray(points, dtype=float)
        metrics = list(metrics)

        if X.ndim != 2:
            raise ValueError("points deve ter shape (n_points, n_metrics).")

        if X.shape[0] == 0:
            raise ValueError("points deve conter pelo menos um ponto.")

        if len(metrics) != X.shape[1]:
            raise ValueError(
                "O número de métricas deve ser igual ao número de colunas."
            )

        metric_names = [metric.name for metric in metrics]
        directions = [metric.direction for metric in metrics]

        if len(metric_names) != len(set(metric_names)):
            raise ValueError(
                "As métricas usadas na seleção devem possuir nomes únicos."
            )

        if any(direction not in {"min", "max"} for direction in directions):
            raise ValueError("Cada direção deve ser 'min' ou 'max'.")

        unknown_targets = set(self.targets) - set(metric_names)
        if unknown_targets:
            raise ValueError(
                "Alvos definidos para métricas inexistentes: "
                f"{sorted(unknown_targets)}."
            )

        if self.weights is not None:
            unknown_weights = set(self.weights) - set(metric_names)
            if unknown_weights:
                raise ValueError(
                    "Pesos definidos para métricas inexistentes: "
                    f"{sorted(unknown_weights)}."
                )

        target_indices = [
            index
            for index, name in enumerate(metric_names)
            if name in self.targets
        ]
        free_indices = [
            index
            for index, name in enumerate(metric_names)
            if name not in self.targets
        ]

        if self.mode == "match":
            return self._select_match(
                X,
                metric_names,
                directions,
                target_indices,
            )

        if not free_indices:
            raise ValueError(
                "O modo 'optimize' precisa deixar pelo menos uma "
                "métrica fora de targets."
            )

        return self._select_optimize(
            X,
            metric_names,
            directions,
            target_indices,
            free_indices,
        )

    def _select_match(
        self,
        X,
        metric_names,
        directions,
        target_indices,
    ):
        target_names = [metric_names[index] for index in target_indices]
        target_values = np.asarray([
            self.targets[name]
            for name in target_names
        ])
        signs = np.asarray([
            1.0 if directions[index] == "max" else -1.0
            for index in target_indices
        ])

        # Valor positivo: alvo satisfeito.
        # Valor negativo: alvo violado.
        differences = (X[:, target_indices] - target_values) * signs
        weights = self._get_weights(target_names)

        violations = np.maximum(-differences, 0.0) * weights
        acceptable = np.maximum(differences, 0.0) * weights

        violation_distance = self._distance(violations)
        target_distance = self._distance(acceptable)

        # Primeiro minimiza a violação; depois a distância ao alvo.
        order = np.lexsort((target_distance, violation_distance))

        return int(order[0])

    def _select_optimize(
        self,
        X,
        metric_names,
        directions,
        target_indices,
        free_indices,
    ):
        candidate_indices = np.arange(X.shape[0])

        if target_indices:
            target_names = [
                metric_names[index]
                for index in target_indices
            ]
            target_values = np.asarray([
                self.targets[name]
                for name in target_names
            ])
            target_signs = np.asarray([
                1.0 if directions[index] == "max" else -1.0
                for index in target_indices
            ])

            differences = (
                X[:, target_indices] - target_values
            ) * target_signs

            feasible = np.all(differences >= 0.0, axis=1)

            if feasible.any():
                candidate_indices = np.flatnonzero(feasible)
            else:
                # Sem solução viável: usa os pontos de menor violação.
                target_weights = self._get_weights(target_names)
                violations = (
                    np.maximum(-differences, 0.0)
                    * target_weights
                )
                violation_distance = self._distance(violations)

                candidate_indices = np.flatnonzero(
                    np.isclose(
                        violation_distance,
                        violation_distance.min(),
                    )
                )

        free_names = [metric_names[index] for index in free_indices]
        free_signs = np.asarray([
            1.0 if directions[index] == "max" else -1.0
            for index in free_indices
        ])

        oriented_values = (
            X[candidate_indices][:, free_indices]
            * free_signs
        )

        ideal_point = oriented_values.max(axis=0)
        weights = self._get_weights(free_names)

        distances = self._distance(
            (ideal_point - oriented_values) * weights
        )

        best_local_index = int(np.argmin(distances))

        return int(candidate_indices[best_local_index])
