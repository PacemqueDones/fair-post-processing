import numpy as np

class TopsisSelector:
    def __init__(self, weights=None):
        self.weights = weights

    def select(self, points, directions):
        X = np.array(points, dtype=float)

        weights = np.ones(X.shape[1]) if self.weights is None else np.array(self.weights, dtype=float)

        if len(weights) != X.shape[1]:
            raise ValueError("Número de pesos deve ser igual ao número de objetivos.")

        # normalização vetorial por coluna
        norm = np.linalg.norm(X, axis=0)
        norm[norm == 0] = 1.0
        Xn = X / norm

        # aplicação dos pesos
        V = Xn * weights

        ideal_best = []
        ideal_worst = []

        for j, direction in enumerate(directions):
            col = V[:, j]

            if direction == "max":
                ideal_best.append(col.max())
                ideal_worst.append(col.min())
            elif direction == "min":
                ideal_best.append(col.min())
                ideal_worst.append(col.max())
            else:
                raise ValueError(f"Direção inválida: {direction}")

        ideal_best = np.array(ideal_best)
        ideal_worst = np.array(ideal_worst)

        d_best = np.linalg.norm(V - ideal_best, axis=1)
        d_worst = np.linalg.norm(V - ideal_worst, axis=1)

        closeness = d_worst / (d_best + d_worst + 1e-12)

        return int(np.argmax(closeness))

class ZenithSelector:
    def __init__(self, weights=None):
        self.weights = weights
    def select(self, points, directions):
        X = np.array(points, dtype=float)
        self.weights = np.ones(X.shape[1]) if self.weights is None else np.array(self.weights)

        # min-max por coluna
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        denom = maxs - mins
        denom[denom == 0] = 1.0
        Xn = (X - mins) / denom

        zenith = []
        for j, direction in enumerate(directions):
            if direction == "max":
                zenith.append(1.0)
            elif direction == "min":
                zenith.append(0.0)
            else:
                raise ValueError(f"Direção inválida: {direction}")

        zenith = np.array(zenith)

        distances = np.linalg.norm((Xn - zenith) * self.weights, axis=1)
        return int(np.argmin(distances))