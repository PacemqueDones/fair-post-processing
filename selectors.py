import numpy as np

class TopsisSelector:
    def select(self, points):
        points = np.array(points)
        ideal = points.min(axis=0)
        nadir = points.max(axis=0)

        d_pos = np.linalg.norm(points - ideal, axis=1)
        d_neg = np.linalg.norm(points - nadir, axis=1)

        score = d_neg / (d_pos + d_neg + 1e-12)
        return np.argmax(score)
    
class ZenithSelector:
    def select(self, points):
        points = np.array(points)
        ideal = points.min(axis=0)
        dist = np.linalg.norm(points - ideal, axis=1)
        return np.argmin(dist)