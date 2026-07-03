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

    def __init__(self, D_X, L_const=1.0, block_size=10000):
        self.D_X = torch.as_tensor(D_X, dtype=torch.float32)
        self.L_const = L_const
        self.block_size = block_size

    def __call__(self, y_true, y_pred, sensitive_attr=None, logits=None):
        if logits is None:
            raise ValueError("IFV rate precisa receber logits.")

        probabilities = torch.softmax(logits, dim=1)
        device = probabilities.device
        dtype = probabilities.dtype

        D_X = self.D_X.to(device=device, dtype=dtype)
        num_samples = probabilities.shape[0]

        if D_X.shape != (num_samples, num_samples):
            raise ValueError(
                f"D_X deve ter shape {(num_samples, num_samples)}."
            )

        total_violations = torch.tensor(0.0, device=device, dtype=dtype)
        total_pairs = 0

        for start in range(0, num_samples, self.block_size):
            end = min(start + self.block_size, num_samples)

            probabilities_block = probabilities[start:end]
            prediction_distance = torch.cdist(
                probabilities_block,
                probabilities,
                p=1,
            ) / 2.0

            violation = (
                prediction_distance
                > self.L_const * D_X[start:end]
            )

            rows = torch.arange(end - start, device=device)
            columns = torch.arange(start, end, device=device)
            violation[rows, columns] = False

            total_violations += violation.float().sum()
            total_pairs += (end - start) * (num_samples - 1)

        return (total_violations / total_pairs).item()
