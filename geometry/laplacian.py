import torch


class LaplacianWeightBuilder:
    """Transforma distâncias brutas em pesos laplacianos.

    Não calcula distâncias.

    Recebe distâncias já calculadas pela
    FairDistanceMetric.
    """

    def __init__(
        self,
        theta=1.0,
        tau_quantile=0.35,
        dtype=torch.float32,
    ):
        if not 0.0 < tau_quantile <= 1.0:
            raise ValueError(
                "tau_quantile deve pertencer "
                "ao intervalo (0, 1]."
            )

        self.theta = theta
        self.tau_quantile = tau_quantile
        self.dtype = dtype

        self.tau_ = None

    def fit(
        self,
        raw_distances,
    ):
        """Calcula o tau a partir das distâncias do treino."""

        raw_distances = torch.as_tensor(
            raw_distances,
            dtype=torch.float64,
        )

        positive_distances = raw_distances[
            raw_distances > 0
        ]

        if positive_distances.numel() == 0:
            raise ValueError(
                "Não existem distâncias positivas "
                "para calcular tau."
            )

        self.tau_ = float(
            torch.quantile(
                positive_distances,
                self.tau_quantile,
            ).item()
        )

        return self

    def transform(
        self,
        raw_distances,
    ):
        """Produz os pesos laplacianos."""

        if self.tau_ is None:
            raise RuntimeError(
                "LaplacianWeightBuilder precisa "
                "ser ajustado com fit()."
            )

        raw_distances = torch.as_tensor(
            raw_distances,
            dtype=self.dtype,
        )

        weights = torch.exp(
            -self.theta
            * raw_distances.pow(2)
        )

        weights = torch.where(
            raw_distances <= self.tau_,
            weights,
            torch.zeros_like(weights),
        )

        return weights