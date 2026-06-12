import torch
from .metric import Metric

class AccuracyMetric(Metric):
    name = "acc"
    direction = "max"   # importante para TOPSIS/Pareto depois
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None, scores=None):
        return (y_pred == y_true).float().mean().item()

class BalancedAccuracyMetric(Metric):
    name = "bacc"
    direction = "max"
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None, scores=None):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        classes = torch.unique(y_true)

        recalls = []

        for c in classes:
            mask = y_true == c

            # Recall da classe c:
            # entre todos os exemplos verdadeiros da classe c,
            # quantos foram previstos corretamente como c?
            recall_c = (y_pred[mask] == c).float().mean()

            recalls.append(recall_c)

        return torch.stack(recalls).mean().item()
    
class PrecisionMetric(Metric):
    name = "prec"
    direction = "max"
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None, scores=None):
        pred_pos = (y_pred == 1)

        if pred_pos.sum() == 0:
            return 0.0

        tp = ((y_pred == 1) & (y_true == 1)).sum().float()
        pp = pred_pos.sum().float()

        precision = tp / pp
        return precision.item()

class RecallMetric(Metric):
    name = "rec"
    direction = "max"
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None, scores=None):
        positive = (y_true == 1)
        if positive.sum() == 0:
            return 0.0
        return y_pred[positive].float().mean().item()
    
class F1ScoreMetric(Metric):
    name = "f1"
    direction = "max"
    type = "performance"

    def __call__(self, y_true, y_pred, sensitive_attr=None, scores=None):
        tp = ((y_pred == 1) & (y_true == 1)).sum().float()
        fp = ((y_pred == 1) & (y_true == 0)).sum().float()
        fn = ((y_pred == 0) & (y_true == 1)).sum().float()

        precision_denom = tp + fp
        recall_denom = tp + fn

        if precision_denom == 0 or recall_denom == 0:
            return 0.0

        precision = tp / precision_denom
        recall = tp / recall_denom

        denom = precision + recall
        if denom == 0:
            return 0.0

        f1 = 2 * (precision * recall) / denom
        return f1.item()

class DemographicParityMetric(Metric):
    name = "ddp"
    direction = "min"
    type = "fairness"

    def __call__(self, y_true, y_pred, sensitive_attr=None, scores=None):
        g0 = (sensitive_attr == 0)
        g1 = (sensitive_attr == 1)

        if g0.sum() == 0 or g1.sum() == 0:
            return 0.0

        rate0 = y_pred[g0].float().mean()
        rate1 = y_pred[g1].float().mean()
        return torch.abs(rate0 - rate1).item()
    
class EqualityOpportunityMetric(Metric):
    name = "deo"
    direction = "min"
    type = "fairness"

    def __call__(self, y_true, y_pred, sensitive_attr=None, scores=None):
        g0 = (sensitive_attr == 0) & (y_true == 1)
        g1 = (sensitive_attr == 1) & (y_true == 1)

        if g0.sum() == 0 or g1.sum() == 0:
            return 0.0

        tpr0 = y_pred[g0].float().mean()
        tpr1 = y_pred[g1].float().mean()
        return torch.abs(tpr0 - tpr1).item()
    
class IndividualFairnessViolationMeanMetric(Metric):
    name = "ifv_mean"
    direction = "min"
    type = "individual_fairness"

    def __init__(self, D_X, L_const=1.0, block_size=10000):
        """
        Métrica baseada na condição de Fairness Through Awareness:

            D(M(x_i), M(x_j)) <= L * d(x_i, x_j)

        Aqui:
        - D é a distância de variação total entre distribuições preditas;
        - D_X é a matriz [n, n] com as distâncias justas entre indivíduos;
        - scores deve ser a saída do modelo antes do softmax.
        """

        self.D_X = torch.as_tensor(D_X, dtype=torch.float32)
        self.L_const = L_const
        self.block_size = block_size

    def __call__(self, y_true, y_pred, sensitive_attr=None, scores=None):

        P = torch.softmax(scores, dim=1)

        device = P.device
        dtype = P.dtype

        D_X = self.D_X.to(device=device, dtype=dtype)

        n = P.shape[0]

        if D_X.shape != (n, n):
            raise ValueError(
                f"D_X deve ter shape {(n, n)}, mas recebeu {tuple(D_X.shape)}."
            )

        total_violation = torch.tensor(0.0, device=device, dtype=dtype)
        total_count = 0

        for start in range(0, n, self.block_size):
            end = min(start + self.block_size, n)

            P_block = P[start:end]

            # D_tv(P_i, P_j) = 1/2 * ||P_i - P_j||_1
            D_pred = torch.cdist(P_block, P, p=1) / 2.0

            # V_ij = max(0, D_tv(P_i, P_j) - L * d_X(i,j))
            violation = torch.relu(
                D_pred - self.L_const * D_X[start:end, :]
            )

            # Remove a diagonal correspondente ao bloco
            rows = torch.arange(end - start, device=device)
            cols = torch.arange(start, end, device=device)
            violation[rows, cols] = 0.0

            total_violation += violation.sum()
            total_count += (end - start) * n - (end - start)

        return (total_violation / total_count).item()


class IndividualFairnessViolationRateMetric(Metric):
    name = "ifv_rate"
    direction = "min"
    type = "individual_fairness"

    def __init__(self, D_X, L_const=1.0, block_size=10000):
        """
        Taxa de pares que violam a condição:

            D(M(x_i), M(x_j)) <= L * d(x_i, x_j)

        Retorna a proporção de pares (i,j), i != j, com violação positiva.
        """

        self.D_X = torch.as_tensor(D_X, dtype=torch.float32)
        self.L_const = L_const
        self.block_size = block_size

    def __call__(self, y_true, y_pred, sensitive_attr=None, scores=None):
        if scores is None:
            raise ValueError(
                "IndividualFairnessViolationRateMetric precisa receber scores=logits_eval."
            )

        P = torch.softmax(scores, dim=1)

        device = P.device
        dtype = P.dtype

        D_X = self.D_X.to(device=device, dtype=dtype)

        n = P.shape[0]

        if D_X.shape != (n, n):
            raise ValueError(
                f"D_X deve ter shape {(n, n)}, mas recebeu {tuple(D_X.shape)}."
            )

        total_violations = torch.tensor(0.0, device=device, dtype=dtype)
        total_count = 0

        for start in range(0, n, self.block_size):
            end = min(start + self.block_size, n)

            P_block = P[start:end]

            # D_tv(P_i, P_j) = 1/2 * ||P_i - P_j||_1
            D_pred = torch.cdist(P_block, P, p=1) / 2.0

            # V_ij = max(0, D_tv(P_i, P_j) - L * d_X(i,j))
            violation = torch.relu(
                D_pred - self.L_const * D_X[start:end, :]
            )

            # Remove a diagonal correspondente ao bloco
            rows = torch.arange(end - start, device=device)
            cols = torch.arange(start, end, device=device)
            violation[rows, cols] = 0.0

            total_violations += (violation > 0).float().sum()
            total_count += (end - start) * n - (end - start)

        return (total_violations / total_count).item()