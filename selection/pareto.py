import numpy as np


def pareto_front(points, directions):
    """
    Retorna os índices dos pontos não-dominados (fronteira de Pareto).

    Parameters
    ----------
    points : array-like of shape (n_points, n_criteria)
        Matriz de pontos.
    directions : list[str]
        Lista com "max" ou "min" para cada critério.

    Returns
    -------
    np.ndarray
        Índices dos pontos que pertencem à fronteira de Pareto.
    """
    X = np.asarray(points, dtype=float)

    if X.ndim != 2:
        raise ValueError("points deve ter shape (n_points, n_criteria).")

    if len(directions) != X.shape[1]:
        raise ValueError(
            "directions deve ter o mesmo número de elementos que o número de critérios."
        )

    signs = np.array([
        1.0 if d == "max" else -1.0 if d == "min" else np.nan
        for d in directions
    ])

    if np.isnan(signs).any():
        raise ValueError("Cada direção deve ser 'max' ou 'min'.")

    X_adj = X * signs

    at_least_as_good = np.all(
        X_adj[None, :, :] >= X_adj[:, None, :],
        axis=2
    )

    strictly_better = np.any(
        X_adj[None, :, :] > X_adj[:, None, :],
        axis=2
    )

    dominated = at_least_as_good & strictly_better
    is_pareto = ~np.any(dominated, axis=1)

    return np.flatnonzero(is_pareto)