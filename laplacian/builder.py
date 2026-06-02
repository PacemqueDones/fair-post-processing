import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform


class MahalanobisLaplacianBuilder:
    def __init__(
        self,
        theta=1.0,
        tau_quantile=0.25,
        laplacian_type="unnormalized",
        eps=1e-8,
        dtype=torch.float32,
        device="cpu",
    ):
        self.theta = theta
        self.tau_quantile = tau_quantile
        self.laplacian_type = laplacian_type
        self.eps = eps
        self.dtype = dtype
        self.device = device

    def _to_numpy(self, X):
        if isinstance(X, torch.Tensor):
            return X.detach().cpu().numpy()

        if hasattr(X, "to_numpy"):
            return X.to_numpy()

        return np.asarray(X)

    def _build_W(self, X):
        X = self._to_numpy(X).astype(np.float64)

        cov = np.cov(X, rowvar=False)
        cov = cov + self.eps * np.eye(cov.shape[0])

        inv_cov = np.linalg.inv(cov)

        dist_vector = pdist(X, metric="mahalanobis", VI=inv_cov)

        tau = np.quantile(dist_vector[dist_vector > 0], self.tau_quantile)

        W_vector = np.exp(-self.theta * dist_vector**2)
        W_vector[dist_vector > tau] = 0.0

        W = squareform(W_vector)

        # Garante simetria
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

        # L = I - D_tilde^{-1} W_tilde
        # Então fora da diagonal: - W_tilde_ij / degree_tilde_i
        L = L / (degree_tilde[:, None] + self.eps)

        np.fill_diagonal(L, 1.0)

        return L

    def build(self, X, return_artifacts=False):
        W = self._build_W(X)

        if self.laplacian_type == "unnormalized":
            L = self._unnormalized_laplacian(W)

        elif self.laplacian_type == "normalized_random_walk":
            L = self._normalized_random_walk_laplacian(W)

        else:
            raise ValueError(
                "laplacian_type deve ser 'unnormalized' "
                "ou 'normalized_random_walk'."
            )

        L_torch = torch.tensor(
            L,
            dtype=self.dtype,
            device=self.device
        )

        if return_artifacts:
            return {
                "W": W,
                "L": L_torch,
                "laplacian_type": self.laplacian_type,
            }

        return L_torch