import numpy as np

class TopsisSelector:
    def select(self, points, directions):
        X = np.array(points, dtype=float)

        norm = np.linalg.norm(X, axis=0)
        norm[norm == 0] = 1.0
        Xn = X / norm

        ideal_best = []
        ideal_worst = []

        for j, direction in enumerate(directions):
            col = Xn[:, j]

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

        d_best = np.linalg.norm(Xn - ideal_best, axis=1)
        d_worst = np.linalg.norm(Xn - ideal_worst, axis=1)

        closeness = d_worst / (d_best + d_worst + 1e-12)

        return int(np.argmax(closeness))

class ZenithSelector:
    def select(self, points, directions):
        X = np.array(points, dtype=float)

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

        distances = np.linalg.norm(Xn - zenith, axis=1)
        return int(np.argmin(distances))
    

class LossTopsisSelector:
    def select(self, points):
        points = np.array(points)
        ideal = points.min(axis=0)
        nadir = points.max(axis=0)

        d_pos = np.linalg.norm(points - ideal, axis=1)
        d_neg = np.linalg.norm(points - nadir, axis=1)

        score = d_neg / (d_pos + d_neg + 1e-12)
        return np.argmax(score)
    
class LossZenithSelector:
    def select(self, points):
        points = np.array(points)
        ideal = points.min(axis=0)
        dist = np.linalg.norm(points - ideal, axis=1)
        return np.argmin(dist)