import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform


class FairLaplacianBuilder:
    def __init__(
        self,
        metric="mahalanobis",
        theta=1.0,
        tau_quantile=0.25,
        laplacian_type="_normalized_random_walk_laplacian",
        eps=1e-8,
        dtype=torch.float32,
    ):
        self.metric = metric
        self.theta = theta
        self.tau_quantile = tau_quantile
        self.laplacian_type = laplacian_type
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

    def _build_W(self, dist_vector):
        positive_distances = dist_vector[dist_vector > 0]

        tau = np.quantile(
            positive_distances,
            self.tau_quantile
        )

        W_vector = np.exp(-self.theta * dist_vector**2)
        W_vector[dist_vector > tau] = 0.0

        W = squareform(W_vector)

        W = 0.5 * (W + W.T)
        np.fill_diagonal(W, 0.0)

        return W

    def _unnormalized_laplacian(self, W):
        degree = W.sum(axis=1)

        L = -W.copy()
        np.fill_diagonal(L, degree)

        return L

    def _normalized_random_walk_laplacian(self, W):
        degree = W.sum(axis=1)

        inv_sqrt_degree = 1.0 / np.sqrt(degree + self.eps)

        W_tilde = (
            inv_sqrt_degree[:, None]
            * W
            * inv_sqrt_degree[None, :]
        )

        degree_tilde = W_tilde.sum(axis=1)

        L = -W_tilde.copy()
        L = L / (degree_tilde[:, None] + self.eps)

        np.fill_diagonal(L, 1.0)

        return L

    def _build_laplacian(self, W):
        if self.laplacian_type == "unnormalized":
            return self._unnormalized_laplacian(W)

        if self.laplacian_type == "normalized_random_walk":
            return self._normalized_random_walk_laplacian(W)

        raise ValueError(
            "laplacian_type deve ser 'unnormalized' "
            "ou 'normalized_random_walk'."
        )

    def build(self, X):
        dist_vector = self._distance_vector(X)

        W = self._build_W(dist_vector)

        L = self._build_laplacian(W)

        return torch.tensor(
            L,
            dtype=self.dtype
        )