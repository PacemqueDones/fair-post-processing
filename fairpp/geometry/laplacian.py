import torch
import numpy as np

from scipy.spatial.distance import pdist, squareform

class FairLaplacianBuilder:
    """Constrói W e a Laplaciana densa.

    Para cada par:

        W_ij = exp(-theta * d_ij²)

    quando:

        d_ij <= tau

    e zero caso contrário.
    """
    
    def __init__(
        self,
        theta=1.0,
        tau_quantile=0.3,
        laplacian_type="unnormalized",
        eps=1e-8,
        dtype=torch.float32,
    ):
        
        valid_laplacians = {"unnormalized", "normalized_random_walk"}


        if not 0.0 < tau_quantile <= 1.0:
            raise ValueError(
                "tau_quantile deve pertencer "
                "ao intervalo (0, 1]."
            )

        if laplacian_type not in valid_laplacians:
            raise ValueError(
                "laplacian_type deve ser "
                "'unnormalized' ou "
                "'normalized_random_walk'."
            )
        
        self.theta = theta
        self.tau_quantile = tau_quantile
        self.laplacian_type = laplacian_type
        self.eps = eps
        self.dtype = dtype

        self.tau_ = None
        self.W_ = None


    def _to_numpy(self, X):
        if isinstance(X, torch.Tensor):
            return X.detach().cpu().numpy()

        if hasattr(X, "to_numpy"):
            return X.to_numpy()

        return np.asarray(X)

    def _build_W(self, distance_vector):
        positive_distances = distance_vector[distance_vector > 0]

        if positive_distances.size == 0:
            raise ValueError(
                "Não existem distâncias "
                "positivas para calcular tau."
            )

        self.tau_ = float(np.quantile(positive_distances, self.tau_quantile))

        W_vector = np.exp(-self.theta * distance_vector**2)

        W_vector[distance_vector > self.tau_] = 0.0

        return W_vector

    def _unnormalized_laplacian(self, W):
        degree = W.sum(axis=1)

        D = np.diag(degree)

        return D - W

    def _normalized_random_walk_laplacian(self, W):
        degree = W.sum(axis=1)

        inverse_sqrt_degree = np.zeros_like(degree, dtype=np.float64)

        valid = degree > self.eps

        inverse_sqrt_degree[valid] = 1.0 / np.sqrt(degree[valid])

        W_normalized = (inverse_sqrt_degree[:, None] * W * inverse_sqrt_degree[None, :])

        normalized_degree = (W_normalized.sum(axis=1))

        inverse_normalized_degree = np.zeros_like(normalized_degree, dtype=np.float64)

        valid_normalized = (normalized_degree > self.eps)

        inverse_normalized_degree[valid_normalized] = (1.0 / normalized_degree[valid_normalized])

        transition = (inverse_normalized_degree[:, None,] * W_normalized)

        L = (np.eye(W.shape[0], dtype=np.float64,) - transition)

        return L

    def _build_laplacian(self, W_vector):

        W = squareform(W_vector)

        np.fill_diagonal(W, 0.0)

        if self.laplacian_type == "unnormalized":
            return self._unnormalized_laplacian(W)

        return self._normalized_random_walk_laplacian(W)

    def build(self, D_vector):
        """Constrói e retorna a matriz Laplaciana."""

        self.W_vector = self._build_W(self._to_numpy(D_vector))

        L = self._build_laplacian(self.W_vector)

        return torch.as_tensor(self.W_vector, dtype=self.dtype), torch.as_tensor(L, dtype=self.dtype)


class LaplacianWeightBuilder:
    """Transforma distâncias brutas em pesos laplacianos.

    Não calcula distâncias.

    Recebe distâncias já calculadas pela
    FairDistanceMetric.
    """

    def __init__(self, theta=1.0, tau_quantile=0.35, dtype=torch.float32):
        if not 0.0 < tau_quantile <= 1.0:
            raise ValueError(
                "tau_quantile deve pertencer "
                "ao intervalo (0, 1]."
            )

        self.theta = theta
        self.tau_quantile = tau_quantile
        self.dtype = dtype

        self.tau_ = None

    def fit(self, raw_distances):
        """Calcula o tau a partir das distâncias do treino."""

        raw_distances = torch.as_tensor(raw_distances, dtype=torch.float64)

        positive_distances = raw_distances[raw_distances > 0]

        if positive_distances.numel() == 0:
            raise ValueError(
                "Não existem distâncias positivas "
                "para calcular tau."
            )

        self.tau_ = float(torch.quantile(positive_distances, self.tau_quantile).item())

        return self

    def transform(self, raw_distances):
        """Produz os pesos laplacianos."""

        if self.tau_ is None:
            raise RuntimeError(
                "LaplacianWeightBuilder precisa "
                "ser ajustado com fit()."
            )

        raw_distances = torch.as_tensor(raw_distances, dtype=self.dtype)

        weights = torch.exp(-self.theta * raw_distances.pow(2))

        weights = torch.where(raw_distances <= self.tau_, weights, torch.zeros_like(weights))

        return weights