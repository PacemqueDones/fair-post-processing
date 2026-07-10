import numpy as np
import torch


class FairDistanceMetric:
    """Calcula distâncias somente para pares fornecidos.

    A distância bruta e a distância normalizada são
    operações separadas.

    A distância bruta é usada pela Laplaciana.

    A distância normalizada é usada pelas métricas IFV.
    """

    def __init__(
        self,
        metric="mahalanobis",
        normalization="fraction",
        theta=1.0,
        fraction_scale=1.0,
        eps=1e-8,
        block_size=100_000,
        dtype=torch.float32,
    ):
        valid_metrics = {
            "mahalanobis",
            "euclidean",
            "manhattan",
        }

        valid_normalizations = {
            "none",
            "minmax",
            "exp",
            "fraction",
        }

        if metric not in valid_metrics:
            raise ValueError(
                "metric deve ser 'mahalanobis', "
                "'euclidean' ou 'manhattan'."
            )

        if normalization not in valid_normalizations:
            raise ValueError(
                "normalization deve ser 'none', "
                "'minmax', 'exp' ou 'fraction'."
            )

        self.metric = metric
        self.normalization = normalization

        self.theta = theta
        self.fraction_scale = fraction_scale
        self.eps = eps
        self.block_size = block_size
        self.dtype = dtype

        self.inverse_covariance_ = None
        self.max_distance_ = None
        self.is_fitted_ = False

    def _to_numpy(self, X):
        if isinstance(X, torch.Tensor):
            return (
                X.detach()
                .cpu()
                .numpy()
            )

        if hasattr(X, "to_numpy"):
            return X.to_numpy()

        return np.asarray(X)

    def fit(self, X):
        """Ajusta a métrica usando o conjunto de treino."""

        X = self._to_numpy(X)

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        if X.ndim != 2:
            raise ValueError(
                "X deve possuir shape "
                "(n_samples, n_features)."
            )

        if self.metric == "mahalanobis":
            covariance = np.cov(
                X,
                rowvar=False,
            )

            covariance = (
                covariance
                + self.eps
                * np.eye(
                    covariance.shape[0]
                )
            )

            self.inverse_covariance_ = (
                np.linalg.pinv(
                    covariance
                )
            )

        self.is_fitted_ = True

        return self

    def compute_pairs(
        self,
        X,
        pair_index,
    ):
        """Calcula as distâncias brutas dos pares."""

        if not self.is_fitted_:
            raise RuntimeError(
                "FairDistanceMetric precisa ser "
                "ajustada com fit(X_train)."
            )

        X = self._to_numpy(X)

        X = np.asarray(
            X,
            dtype=np.float64,
        )

        pair_index = torch.as_tensor(
            pair_index,
            dtype=torch.long,
        )

        pair_index = (
            pair_index
            .detach()
            .cpu()
            .numpy()
        )

        if (
            pair_index.ndim != 2
            or pair_index.shape[0] != 2
        ):
            raise ValueError(
                "pair_index deve possuir shape "
                "(2, num_pairs)."
            )

        num_pairs = pair_index.shape[1]

        distances = np.empty(
            num_pairs,
            dtype=np.float64,
        )

        for start in range(
            0,
            num_pairs,
            self.block_size,
        ):
            end = min(
                start + self.block_size,
                num_pairs,
            )

            source = pair_index[
                0,
                start:end,
            ]

            target = pair_index[
                1,
                start:end,
            ]

            differences = (
                X[source]
                - X[target]
            )

            if self.metric == "euclidean":
                block_distances = np.sqrt(
                    np.sum(
                        differences**2,
                        axis=1,
                    )
                )

            elif self.metric == "manhattan":
                block_distances = np.sum(
                    np.abs(differences),
                    axis=1,
                )

            else:
                transformed = (
                    differences
                    @ self.inverse_covariance_
                )

                squared_distances = np.sum(
                    transformed
                    * differences,
                    axis=1,
                )

                squared_distances = np.maximum(
                    squared_distances,
                    0.0,
                )

                block_distances = np.sqrt(
                    squared_distances
                )

            distances[start:end] = (
                block_distances
            )

        return torch.as_tensor(
            distances,
            dtype=self.dtype,
        )

    def fit_normalization(
        self,
        raw_distances,
    ):
        """Ajusta parâmetros da normalização.

        Atualmente necessário apenas para minmax.
        """

        raw_distances = torch.as_tensor(
            raw_distances,
            dtype=torch.float64,
        )

        if self.normalization == "minmax":
            self.max_distance_ = float(
                raw_distances.max().item()
            )

        return self

    def normalize(
        self,
        raw_distances,
    ):
        """Transforma distâncias brutas em distâncias IFV."""

        raw_distances = torch.as_tensor(
            raw_distances,
            dtype=self.dtype,
        )

        if self.normalization == "none":
            return raw_distances.clone()

        if self.normalization == "fraction":
            return (
                raw_distances
                / (
                    self.fraction_scale
                    + raw_distances
                    + self.eps
                )
            )

        if self.normalization == "exp":
            return (
                1.0
                - torch.exp(
                    -self.theta
                    * raw_distances.pow(2)
                )
            )

        if self.max_distance_ is None:
            raise RuntimeError(
                "Para normalization='minmax', "
                "execute fit_normalization usando "
                "distâncias do treino."
            )

        return (
            raw_distances
            / (
                self.max_distance_
                + self.eps
            )
        )