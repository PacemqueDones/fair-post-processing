from .geometry import FairPairGeometry


class FairGeometryBuilder:
    """Coordena amostragem, distâncias e pesos."""

    def __init__(
        self,
        pair_sampler,
        distance_metric,
        laplacian_weight_builder=None,
        tau_max_pairs=500_000,
        tau_random_state=30,
    ):
        self.pair_sampler = pair_sampler
        self.distance_metric = distance_metric
        self.laplacian_weight_builder = (
            laplacian_weight_builder
        )

        self.tau_max_pairs = tau_max_pairs
        self.tau_random_state = (
            tau_random_state
        )

        self.is_fitted_ = False

    def fit(self, X_train):
        """Ajusta toda a geometria usando o treino.

        1. Ajusta a métrica de distância.
        2. Seleciona pares para estimar a distribuição
           das distâncias.
        3. Ajusta a normalização.
        4. Ajusta tau da Laplaciana.
        """

        self.distance_metric.fit(
            X_train
        )

        num_samples = len(X_train)

        original_max_pairs = (
            self.pair_sampler.max_pairs
        )

        self.pair_sampler.max_pairs = min(
            original_max_pairs,
            self.tau_max_pairs,
        )

        tau_pair_index = (
            self.pair_sampler.sample(
                num_samples=num_samples,
                random_state=self.tau_random_state,
            )
        )

        self.pair_sampler.max_pairs = (
            original_max_pairs
        )

        tau_raw_distances = (
            self.distance_metric.compute_pairs(
                X_train,
                tau_pair_index,
            )
        )

        self.distance_metric.fit_normalization(
            tau_raw_distances
        )

        if (
            self.laplacian_weight_builder
            is not None
        ):
            self.laplacian_weight_builder.fit(
                tau_raw_distances
            )

        self.is_fitted_ = True

        return self

    def build(
        self,
        X,
        include_laplacian=False,
        random_state=None,
    ):
        """Constrói uma geometria para um conjunto."""

        if not self.is_fitted_:
            raise RuntimeError(
                "FairGeometryBuilder precisa "
                "ser ajustado com fit(X_train)."
            )

        num_samples = len(X)

        num_total_pairs = (
            self.pair_sampler.num_total_pairs(
                num_samples
            )
        )

        pair_index = (
            self.pair_sampler.sample(
                num_samples=num_samples,
                random_state=random_state,
            )
        )

        raw_distances = (
            self.distance_metric.compute_pairs(
                X,
                pair_index,
            )
        )

        fair_distances = (
            self.distance_metric.normalize(
                raw_distances
            )
        )

        edge_weights = None
        tau = None

        if include_laplacian:
            if (
                self.laplacian_weight_builder
                is None
            ):
                raise ValueError(
                    "include_laplacian=True exige "
                    "laplacian_weight_builder."
                )

            edge_weights = (
                self.laplacian_weight_builder
                .transform(
                    raw_distances
                )
            )

            tau = (
                self.laplacian_weight_builder
                .tau_
            )

        num_stored_pairs = (
            pair_index.shape[1]
        )

        return FairPairGeometry(
            pair_index=pair_index,
            raw_distances=raw_distances,
            fair_distances=fair_distances,
            edge_weights=edge_weights,
            tau=tau,
            num_samples=num_samples,
            num_total_pairs=num_total_pairs,
            num_stored_pairs=num_stored_pairs,
            is_exact=(
                num_stored_pairs
                == num_total_pairs
            ),
        )