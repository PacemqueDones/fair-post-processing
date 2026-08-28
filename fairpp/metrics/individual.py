import torch

from .metric import Metric


class IndividualFairnessViolationMeanMetric(Metric):
    name = "ifv_mean"
    direction = "min"
    type = "individual_fairness"

    def __init__(self, D_X, L_const=1.0, block_size=10000):
        self.D_X = torch.as_tensor(D_X, dtype=torch.float32)
        self.L_const = L_const
        self.block_size = block_size

    def __call__(self, y_true, y_pred, sensitive_attr=None, logits=None):
        if logits is None:
            raise ValueError("IFV mean precisa receber logits.")

        probabilities = torch.softmax(logits, dim=1)
        device = probabilities.device
        dtype = probabilities.dtype

        D_X = self.D_X.to(device=device, dtype=dtype)
        num_samples = probabilities.shape[0]

        if D_X.shape != (num_samples, num_samples):
            raise ValueError(
                f"D_X deve ter shape {(num_samples, num_samples)}."
            )

        total_violation = torch.tensor(0.0, device=device, dtype=dtype)
        total_pairs = 0

        for start in range(0, num_samples, self.block_size):
            end = min(start + self.block_size, num_samples)

            probabilities_block = probabilities[start:end]
            prediction_distance = torch.cdist(
                probabilities_block,
                probabilities,
                p=1,
            ) / 2.0

            violation = torch.relu(
                prediction_distance
                - self.L_const * D_X[start:end]
            )

            rows = torch.arange(end - start, device=device)
            columns = torch.arange(start, end, device=device)
            violation[rows, columns] = 0.0

            total_violation += violation.sum()
            total_pairs += (end - start) * (num_samples - 1)

        return (total_violation / total_pairs).item()


class IndividualFairnessViolationRateMetric(Metric):
    name = "ifv_rate"
    direction = "min"
    type = "individual_fairness"

    def __init__(
        self,
        N_vector,
        L_const=1.0,
    ):
        self.N_vector = torch.as_tensor(
            N_vector,
            dtype=torch.float32,
        )

        if self.N_vector.ndim != 1:
            raise ValueError(
                "N_vector deve ser um vetor condensado."
            )

        self.L_const = L_const

    def __call__(
        self,
        y_true,
        y_pred,
        sensitive_attr=None,
        logits=None,
    ):
        if logits is None:
            raise ValueError(
                "IFV rate precisa receber logits."
            )

        probabilities = torch.softmax(logits, dim=1)

        num_samples = probabilities.shape[0]

        expected_pairs = num_samples * (num_samples - 1) // 2

        if self.N_vector.numel() != expected_pairs:
            raise ValueError(
                f"N_vector deveria possuir "
                f"{expected_pairs} distâncias, "
                f"mas possui {self.N_vector.numel()}."
            )

        N_vector = self.N_vector.to(
            device=probabilities.device,
            dtype=probabilities.dtype,
        )

        prediction_distances = torch.pdist(probabilities, p=1) / 2.0

        violations = prediction_distances > self.L_const * N_vector

        return violations.float().mean().item()
    

class ConsistencyScoreMetric(Metric):
    name = "consistency"
    direction = "max"
    type = "individual_fairness"

    def __init__(
        self,
        neighbor_indices,
    ):
        neighbor_indices = torch.as_tensor(neighbor_indices)

        if neighbor_indices.ndim != 2:
            raise ValueError(
                "neighbor_indices deve ter shape "
                "(n_samples, n_neighbors)."
            )

        if torch.is_floating_point(neighbor_indices):
            raise ValueError(
                "ConsistencyScoreMetric agora espera uma matriz "
                "pre-processada de indices de vizinhos, nao X."
            )

        self.neighbor_indices = neighbor_indices.to(dtype=torch.long)

    def __call__(
        self,
        y_true,
        y_pred,
        sensitive_attr=None,
        logits=None,
    ):
        if logits is None:
            raise ValueError(
                "Consistency Score precisa receber logits."
            )

        probabilities = torch.softmax(logits, dim=1)

        neighbor_indices = self.neighbor_indices.to(
            probabilities.device
        )

        num_samples = probabilities.shape[0]

        if neighbor_indices.shape[0] != num_samples:
            raise ValueError(
                "O número de amostras usado para construir "
                "os vizinhos deve ser igual ao número de logits."
            )

        if neighbor_indices.min() < 0 or neighbor_indices.max() >= num_samples:
            raise ValueError(
                "neighbor_indices contem indices fora do intervalo "
                "dos logits recebidos."
            )

        neighbor_probabilities = probabilities[
            neighbor_indices
        ]

        mean_neighbor_probabilities = (
            neighbor_probabilities.mean(dim=1)
        )

        individual_inconsistency = (
            0.5
            * torch.abs(
                probabilities
                - mean_neighbor_probabilities
            ).sum(dim=1)
        )

        mean_inconsistency = (
            individual_inconsistency.mean()
        )

        consistency = 1.0 - mean_inconsistency

        return consistency.item()


class SampledIndividualFairnessViolationRateMetric(
    Metric
):
    name = "ifv_rate"
    direction = "min"
    type = "individual_fairness"

    def __init__(
        self,
        geometry,
        L_const=1.0,
        pair_block_size=100_000,
    ):
        self.geometry = geometry
        self.L_const = L_const
        self.pair_block_size = (
            pair_block_size
        )

    def __call__(
        self,
        y_true,
        y_pred,
        sensitive_attr=None,
        logits=None,
    ):
        if logits is None:
            raise ValueError(
                "IFV rate precisa receber logits."
            )

        probabilities = torch.softmax(logits, dim=1)

        geometry = self.geometry

        if probabilities.shape[0] != geometry.num_samples:
            raise ValueError(
                "O número de probabilidades deve "
                "ser igual ao número de amostras "
                "da geometria."
            )

        pair_index = geometry.pair_index.to(probabilities.device)

        fair_distances = geometry.fair_distances.to(
                device=probabilities.device,
                dtype=probabilities.dtype,
            )
        

        total_violations = torch.tensor(
            0.0,
            device=probabilities.device,
            dtype=probabilities.dtype,
        )

        num_pairs = (
            geometry.num_stored_pairs
        )

        for start in range(
            0,
            num_pairs,
            self.pair_block_size,
        ):
            end = min(start + self.pair_block_size, num_pairs)

            source = pair_index[0, start:end]

            target = pair_index[1, start:end]

            prediction_distance = (
                0.5
                * torch.abs(
                    probabilities[source]
                    - probabilities[target]
                ).sum(dim=1)
            )

            violations = (
                prediction_distance
                > self.L_const
                * fair_distances[
                    start:end
                ]
            )

            total_violations = (
                total_violations
                + violations.float().sum()
            )

        return (
            total_violations
            / num_pairs
        ).item()