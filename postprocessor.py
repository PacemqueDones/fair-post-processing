import torch

from .selection.pareto import pareto_front
from .optimization import build_aggregator
from torchjd.autojac import backward as jacobian_backward
from torchjd.autojac import jac_to_grad


class FairPostProcessor:
    def __init__(
        self,
        model,
        objectives,
        selector,
        selection_metrics: list | None = None,
        aggregator: str = "upgrad",
        normalize_objectives: bool = False,
        lr: float = 1e-2,
        epochs: int = 100,
    ):

        self.model = model
        self.objectives = objectives
        self.selector = selector
        self.selection_metrics = selection_metrics or []
        self.aggregator_name = aggregator
        self.aggregator_ = build_aggregator(aggregator)
        self.normalize_objectives = normalize_objectives
        self.lr = lr
        self.epochs = epochs

        # Histórico completo do treinamento.
        # Cada posição de history_ deve representar uma época.
        self.history_ = []

        # Informações da seleção final
        self.pareto_front_ = []
        self.pareto_indices_ = []
        self.best_index_ = None

        self.best_thresholds_ = None
        self.best_metrics_ = None
        self.best_losses_ = None

        # Metadados das métricas usadas na seleção
        self.metric_names_ = None
        self.metric_directions_ = None

    def _compute_losses(self, logits, y_true, sensitive_attr):
        losses = []
        loss_dict = {}

        for obj in self.objectives:
            loss = obj(logits, y_true, sensitive_attr)
            losses.append(loss)
            loss_dict[obj.name] = loss.item()

        return losses, loss_dict
    
    def _get_trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]


    def fit(self, probs, y_true, sensitive_attr):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        probs = torch.tensor(probs, dtype=torch.float32)
        y_true = torch.tensor(y_true, dtype=torch.long)
        sensitive_attr = torch.tensor(sensitive_attr, dtype=torch.long)

        for epoch in range(self.epochs):
            logits = self.model(probs)

            losses, loss_dict = self._compute_losses(logits, y_true, sensitive_attr)
            params = self._get_trainable_params()

            optimizer.zero_grad()

            loss_vector = torch.stack(losses)

            jacobian_backward(loss_vector)
            jac_to_grad(params, self.aggregator_)

            optimizer.step()

            with torch.no_grad():
                logits_eval = self.model(probs)
                y_pred = torch.argmax(logits_eval, dim=1)

                metric_dict = {}
                point = []

                for metric in self.selection_metrics:
                    value = metric(
                        y_true=y_true,
                        y_pred=y_pred,
                        sensitive_attr=sensitive_attr,
                        scores=logits_eval,
                    )

                    metric_dict[metric.name] = value
                    point.append(value)

                epoch_record = {
                    "epoch": epoch,
                    "losses": loss_dict,
                    "metrics": metric_dict,
                    "point": point,
                    "thresholds": self.model.thresholds.detach().clone(),
                }

                self.history_.append(epoch_record)

        self.metric_names_ = [metric.name for metric in self.selection_metrics]
        self.metric_directions_ = [metric.direction for metric in self.selection_metrics]

        pareto_points = [record["point"] for record in self.history_]

        front_idx = pareto_front(pareto_points, self.metric_directions_)

        self.pareto_indices_ = front_idx
        self.pareto_front_ = [pareto_points[i] for i in front_idx]

        local_idx = self.selector.select(self.pareto_front_, self.metric_directions_)
        global_idx = front_idx[local_idx]

        best_record = self.history_[global_idx]

        self.best_index_ = global_idx
        self.best_thresholds_ = best_record["thresholds"]
        self.best_metrics_ = best_record["metrics"]
        self.best_losses_ = best_record["losses"]

        return self

    def predict(self, probs):
        probs = torch.tensor(probs, dtype=torch.float32)
        with torch.no_grad():
            self.model.thresholds.copy_(self.best_thresholds_)
            logits = self.model(probs)
            return torch.argmax(logits, dim=1).cpu().numpy()
        
    def predict_proba(self, probs):
        probs = torch.tensor(probs, dtype=torch.float32)
        with torch.no_grad():
            self.model.thresholds.copy_(self.best_thresholds_)
            logits = self.model(probs)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def get_thresholds(self):
        return self.best_thresholds_