import torch

from .selection import pareto_front
from .optimization import build_aggregator
from torchjd.autojac import backward as jacobian_backward
from torchjd.autojac import jac_to_grad


class FairPostProcessor:
    def __init__(
        self,
        model,
        objectives,
        selector,
        selection_metrics: list,
        aggregator: str = "upgrad",
        lr: float = 1e-2,
        epochs: int = 100,
        device=None,
    ):

        if model is None:
            raise ValueError("model deve ser fornecido.")

        if not objectives:
            raise ValueError(
                "objectives deve conter ao menos um objetivo."
            )

        if selector is None:
            raise ValueError("selector deve ser fornecido.")

        if not selection_metrics:
            raise ValueError(
                "selection_metrics deve conter ao menos uma métrica."
            )

        self.device = self._resolve_device(device, model)
        self.model = model.to(self.device)
        self.objectives = objectives
        self.selector = selector
        self.selection_metrics = selection_metrics
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


    def _resolve_device(
        self, 
        device, 
        model
    ):
        
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

    def _reset_fit_state(
        self
    ):
        self.history_ = []
        self.pareto_front_ = []
        self.pareto_indices_ = []
        self.best_index_ = None
        self.best_model_state_ = None
        self.best_metrics_ = None
        self.best_losses_ = None
        self.metric_names_ = None
        self.metric_directions_ = None

    def _as_tensor(
        self, 
        values, 
        dtype
    ):
        return torch.as_tensor(values, dtype=dtype, device=self.device)


    def _as_optional_tensor(
        self, 
        values, 
        dtype
    ):
        if values is None:
            return None

        return self._as_tensor(values, dtype=dtype)


    def _prepare_data(
        self,
        inputs,
        y_true=None,
        sensitive_attr=None,
        X=None,
    ):
        inputs = self._as_tensor(inputs, dtype=torch.float32)

        y_true = self._as_optional_tensor(y_true, dtype=torch.long)

        sensitive_attr = self._as_optional_tensor(sensitive_attr, dtype=torch.long)

        X = self._as_optional_tensor(X, dtype=torch.float32)

        return inputs, y_true, sensitive_attr, X


    def _get_trainable_params(
        self
    ):
        return [p for p in self.model.parameters() if p.requires_grad]


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
    

    def _select_best_solution(
        self
    ):
        self.metric_names_ = [metric.name for metric in self.selection_metrics]

        self.metric_directions_ = [metric.direction for metric in self.selection_metrics]

        pareto_points = [record["point"] for record in self.history_]

        self.pareto_indices_ = pareto_front(points=pareto_points, directions=self.metric_directions_)

        self.pareto_front_ = [pareto_points[index] for index in self.pareto_indices_]

        local_index = self.selector.select(self.pareto_front_, self.selection_metrics)

        self.best_index_ = int(self.pareto_indices_[local_index])

        best_record = self.history_[self.best_index_]

        self.best_model_state_ = best_record["model_state"]
        self.best_metrics_ = best_record["metrics"]
        self.best_losses_ = best_record["losses"]

        self.model.load_state_dict(self.best_model_state_)


    def fit(
        self,
        train_inputs,
        train_y_true=None,
        train_sensitive_attr=None,
        train_X=None,
        *,
        val_inputs,
        val_y_true=None,
        val_sensitive_attr=None,
        val_X=None,
    ):
        self.model.to(self.device)

        params = self._get_trainable_params()
        optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9, nesterov=True,)

        train_inputs, train_y_true, train_sensitive_attr, train_X = (
            self._prepare_data(inputs=train_inputs, y_true=train_y_true, sensitive_attr=train_sensitive_attr,X=train_X)
            )

        val_inputs, val_y_true, val_sensitive_attr, val_X = (
            self._prepare_data(inputs=val_inputs, y_true=val_y_true, sensitive_attr=val_sensitive_attr,X=val_X)
            )

        self._reset_fit_state()

        for epoch in range(self.epochs):
            self.model.train()

            train_logits = self.model(inputs=train_inputs, sensitive_attr=train_sensitive_attr, X=train_X)

            loss_vector, loss_dict = self._compute_losses(
                train_logits,
                train_y_true,
                train_sensitive_attr,
            )

            optimizer.zero_grad()

            jacobian_backward(loss_vector)
            jac_to_grad(params, self.aggregator_)

            optimizer.step()

            self.model.eval()

            with torch.no_grad():
                val_logits = self.model(inputs=val_inputs, sensitive_attr=val_sensitive_attr, X=val_X)

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
                    "model_state": { name: tensor.detach().clone().cpu() for name, tensor in self.model.state_dict().items()},
                }

                self.history_.append(epoch_record)

        self._select_best_solution()

        return self

    
    def _check_is_fitted(
        self
    ):
        if self.best_model_state_ is None:
            raise RuntimeError(
                "O FairPostProcessor ainda não foi ajustado. "
                "Execute fit antes de predict ou predict_proba."
            )


    def _predict_logits(
        self,
        inputs,
        sensitive_attr=None,
        X=None,
    ):
        self._check_is_fitted()
        self.model.to(self.device)

        inputs, _, sensitive_attr, X = self._prepare_data(inputs=inputs, sensitive_attr=sensitive_attr, X=X)

        self.model.eval()

        with torch.no_grad():
            self.model.load_state_dict(self.best_model_state_)

            return self.model(inputs=inputs, sensitive_attr=sensitive_attr, X=X)


    def predict(
        self,
        inputs,
        sensitive_attr=None,
        X=None,
    ):
        logits = self._predict_logits(inputs=inputs, sensitive_attr=sensitive_attr, X=X)

        return torch.argmax(logits, dim=1).cpu().numpy()


    def predict_proba(
        self,
        inputs,
        sensitive_attr=None,
        X=None,
    ):
        logits = self._predict_logits(inputs=inputs, sensitive_attr=sensitive_attr, X=X)

        return torch.softmax(logits, dim=1).cpu().numpy()
