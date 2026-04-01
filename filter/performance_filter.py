import numpy as np
import warnings


class PerformanceRangeFilter:
    """
    Filtra os índices dos pontos que respeitam uma margem de tolerância
    nas métricas de desempenho.

    Parameters
    ----------
    alpha : float, default=0.1
        Margem de tolerância.

        Se mode="relative":
            aceita pontos com value >= (1 - alpha) * ref
            para métricas de maximização.

        Se mode="absolute":
            aceita pontos com value >= ref - alpha
            para métricas de maximização.

    mode : {"relative", "absolute"}, default="relative"
        Tipo da margem.

    fallback : {"all", "least_violation"}, default="least_violation"
        O que fazer se nenhum ponto for viável.

        - "all": devolve todos os índices.
        - "least_violation": devolve os índices com menor violação total.
    """

    def __init__(self, alpha=0.1, fallback="least_violation"):
        self.alpha = alpha
        self.fallback = fallback

    def _lower_bound(self, ref):
        return (1.0 - self.alpha) * ref

    def _upper_bound(self, ref):
        return (1.0 + self.alpha) * ref

    def _point_violation(self, metric_dict, metrics, reference_metrics):
        """
        Calcula a violação total de um ponto considerando apenas métricas
        de desempenho com direction='max'.

        Retorna 0.0 se o ponto for viável.
        """
        total_violation = 0.0

        for metric in metrics:
            if metric.type != "performance":
                continue

            value = float(metric_dict[metric.name])
            ref = float(reference_metrics[metric.name])

            if metric.direction == "max":
                lower = self._lower_bound(ref)
                violation = max(0.0, lower - value)
                total_violation += violation

            elif metric.direction == "min":
                upper = self._upper_bound(ref)
                violation = max(0.0, value - upper)
                total_violation += violation

            else:
                raise ValueError(f"Direção inválida: {metric.direction}")

        return total_violation

    def get_indices(self, metric_history, metrics, reference_metrics):
        """
        Retorna os índices dos pontos viáveis.

        Parameters
        ----------
        metric_history : list[dict]
            Lista de dicionários de métricas, um por ponto da fronteira.

        metrics : list
            Lista de objetos de métrica usados para validar as restrições de desempenho.

        reference_metrics : dict
            Dicionário com os valores de referência, por nome de métrica.
            Exemplo: {"acc": 0.84, "f1": 0.67}

        Returns
        -------
        list[int]
            Índices dos pontos viáveis, ou fallback caso nenhum seja viável.
        """
        feasible_indices = []
        violations = []

        for i, metric_dict in enumerate(metric_history):
            violation = self._point_violation(metric_dict, metrics, reference_metrics)
            violations.append(violation)

            if np.isclose(violation, 0.0):
                feasible_indices.append(i)

        if feasible_indices:
            return feasible_indices, {
                "fallback_used": False,
                "fallback_reason": None,
                "min_violation": 0.0,
            }

        if self.fallback == "all":
            warnings.warn(
                "Nenhum ponto da fronte satisfez a restrição de desempenho. "
                "Aplicando fallback='all'.",
                RuntimeWarning
            )
            return list(range(len(metric_history))), {
                "fallback_used": True,
                "fallback_reason": "all",
                "min_violation": float(min(violations)) if violations else None,
            }

        if self.fallback == "least_violation":
            min_violation = min(violations)
            idx = [i for i, v in enumerate(violations) if np.isclose(v, min_violation)]

            warnings.warn(
                "Nenhum ponto da fronte satisfez a restrição de desempenho. "
                "Aplicando fallback='least_violation'.",
                RuntimeWarning
            )

            return idx, {
                "fallback_used": True,
                "fallback_reason": "least_violation",
                "min_violation": float(min_violation),
            }

        raise ValueError("fallback deve ser 'all' ou 'least_violation'")