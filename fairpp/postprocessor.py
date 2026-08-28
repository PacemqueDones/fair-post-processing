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
        lr: float = 1e-2,
        epochs: int = 100,
        device=None,
    ):

        self.device = self._resolve_device(device, model)
        self.model = model.to(self.device)
        self.objectives = objectives
        self.selector = selector
        self.selection_metrics = selection_metrics or []
        self.aggregator_name = aggregator
        self.aggregator_ = build_aggregator(aggregator)
        self.lr = lr
        self.epochs = epochs

        # Histórico completo do treinamento.
        # Cada posição de history_ deve representar uma época.
        self.history_ = []

        # Informações da seleção final
        self.pareto_front_ = []
        self.pareto_indices_ = []
        self.best_index_ = None

        self.best_model_state_ = None
        self.best_metrics_ = None
        self.best_losses_ = None

        # Metadados das métricas usadas na seleção
        self.metric_names_ = None
        self.metric_directions_ = None

    def _resolve_device(self, device, model):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device is None:
            try:
                return next(model.parameters()).device
            except StopIteration:
                return torch.device("cpu")

        resolved_device = torch.device(device)

        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' foi solicitado, mas CUDA nao esta disponivel."
            )

        return resolved_device

    def _as_tensor(self, values, dtype):
        return torch.as_tensor(values, dtype=dtype, device=self.device)

    def _compute_losses(
        self,
        logits,
        y_true,
        sensitive_attr,
    ):
        losses = []
        loss_dict = {}

        for objective in self.objectives:
            objective_losses, objective_names = objective(logits,y_true,sensitive_attr)

            if len(objective_losses) != len(objective_names):
                raise ValueError(
                    f"Objective '{objective.name}' returned "
                    f"{len(objective_losses)} losses but "
                    f"{len(objective_names)} names."
                )

            for loss_name, loss in zip(objective_names,objective_losses):
                if loss.ndim != 0:
                    raise ValueError(
                        f"Loss '{loss_name}' must be a scalar tensor, "
                        f"but received shape {tuple(loss.shape)}."
                    )

                if loss_name in loss_dict:
                    raise ValueError(
                        f"Duplicated loss name: '{loss_name}'."
                    )

                losses.append(loss)
                loss_dict[loss_name] = loss.detach().item()

        if not losses:
            raise ValueError(
                "No losses were returned by the objectives."
            )

        loss_vector = torch.stack(losses)

        return loss_vector, loss_dict
    
    def _get_trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]


    def fit(
        self,
        train_probs,
        train_y_true,
        train_sensitive_attr,
        val_probs,
        val_y_true,
        val_sensitive_attr,
        ):
        self.model.to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        train_probs = self._as_tensor(train_probs, dtype=torch.float32)
        train_y_true = self._as_tensor(train_y_true, dtype=torch.long)
        train_sensitive_attr = self._as_tensor(train_sensitive_attr, dtype=torch.long)

        val_probs = self._as_tensor(val_probs, dtype=torch.float32)
        val_y_true = self._as_tensor(val_y_true, dtype=torch.long)
        val_sensitive_attr = self._as_tensor(val_sensitive_attr, dtype=torch.long)

        for epoch in range(self.epochs):
            train_logits = self.model(train_probs, train_sensitive_attr)

            loss_vector, loss_dict = self._compute_losses(
                train_logits,
                train_y_true,
                train_sensitive_attr,
            )

            params = self._get_trainable_params()

            optimizer.zero_grad()

            jacobian_backward(loss_vector)
            jac_to_grad(params, self.aggregator_)

            optimizer.step()

            with torch.no_grad():
                val_logits = self.model(val_probs, val_sensitive_attr)

                val_y_pred = torch.argmax(val_logits, dim=1)

                metric_dict = {}
                point = []

                for metric in self.selection_metrics:
                    value = metric(
                        y_true=val_y_true,
                        y_pred=val_y_pred,
                        sensitive_attr=val_sensitive_attr,
                        logits=val_logits,
                    )

                    metric_dict[metric.name] = value
                    point.append(value)

                epoch_record = {
                    "epoch": epoch,
                    "losses": loss_dict,
                    "metrics": metric_dict,
                    "point": point,
                    "model_state": {
                        name: tensor.detach().clone().to(self.device)
                        for name, tensor
                        in self.model.state_dict().items()
                    },
                }

                self.history_.append(epoch_record)

        self.metric_names_ = [metric.name for metric in self.selection_metrics]
        self.metric_directions_ = [metric.direction for metric in self.selection_metrics]

        pareto_points = [record["point"] for record in self.history_]

        front_idx = pareto_front(pareto_points, self.metric_directions_)

        self.pareto_indices_ = front_idx
        self.pareto_front_ = [pareto_points[i] for i in front_idx]

        local_idx = self.selector.select(
            self.pareto_front_,
            self.selection_metrics,
        )
        global_idx = front_idx[local_idx]

        best_record = self.history_[global_idx]

        self.best_index_ = global_idx
        self.best_model_state_ = best_record["model_state"]
        self.model.load_state_dict(self.best_model_state_)
        self.best_metrics_ = best_record["metrics"]
        self.best_losses_ = best_record["losses"]

        return self
    
    def _check_is_fitted(self):
        if self.best_model_state_ is None:
            raise RuntimeError(
                "O FairPostProcessor ainda não foi ajustado. "
                "Execute fit antes de predict ou predict_proba."
            )

    def predict(
        self,
        base_probs,
        sensitive_attr=None,
    ):
        self._check_is_fitted()
        self.model.to(self.device)
        base_probs = self._as_tensor(base_probs, dtype=torch.float32)

        if sensitive_attr is not None:
            sensitive_attr = self._as_tensor(sensitive_attr, dtype=torch.long)

        with torch.no_grad():
            self.model.load_state_dict(self.best_model_state_)

            logits = self.model(base_probs, sensitive_attr)

            return torch.argmax(logits, dim=1).cpu().numpy()
        
    def predict_proba(
        self,
        base_probs,
        sensitive_attr=None,
    ):
        self._check_is_fitted()
        self.model.to(self.device)
        base_probs = self._as_tensor(base_probs, dtype=torch.float32)

        if sensitive_attr is not None:
            sensitive_attr = self._as_tensor(sensitive_attr, dtype=torch.long)

        with torch.no_grad():
            self.model.load_state_dict(self.best_model_state_)

            logits = self.model(base_probs,sensitive_attr)

            return torch.softmax(logits,dim=1,).cpu().numpy()
