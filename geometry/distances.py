import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform


class FairDistanceBuilder:
    def __init__(
        self,
        metric="mahalanobis",
        normalization="fraction",
        theta=1.0,
        fraction_scale=1.0,
        eps=1e-8,
        dtype=torch.float32,
    ):
        self.metric = metric
        self.normalization = normalization
        self.theta = theta
        self.fraction_scale = fraction_scale
        self.eps = eps
        self.dtype = dtype

    def _to_numpy(self, X):
        if isinstance(X, torch.Tensor):
            return X.detach().cpu().numpy()

        if hasattr(X, "to_numpy"):
            return X.to_numpy()

        return np.asarray(X)

    def _distance_vector(self, X):
        X = self._to_numpy(X).astype(np.float64)

        if self.metric == "mahalanobis":
            cov = np.cov(X, rowvar=False)
            cov = cov + self.eps * np.eye(cov.shape[0])
            inv_cov = np.linalg.inv(cov)

            return pdist(
                X,
                metric="mahalanobis",
                VI=inv_cov
            )

        if self.metric == "euclidean":
            return pdist(
                X,
                metric="euclidean"
            )

        if self.metric == "manhattan":
            return pdist(
                X,
                metric="cityblock"
            )

        raise ValueError(
            "metric deve ser 'mahalanobis', 'euclidean' ou 'manhattan'."
        )

    def _normalize(self, D):
        if self.normalization == "none":
            return D

        if self.normalization == "minmax":
            return D / (D.max() + self.eps)

        if self.normalization == "exp":
            return 1.0 - np.exp(-self.theta * D**2)

        if self.normalization == "fraction":
            return D / (self.fraction_scale + D + self.eps)

        raise ValueError(
            "normalization deve ser 'none', 'minmax', 'exp' ou 'fraction'."
        )

    def build(self, X):
        dist_vector = self._distance_vector(X)

        D = squareform(dist_vector)

        D = self._normalize(D)

        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)

        return torch.tensor(
            D,
            dtype=self.dtype
        )