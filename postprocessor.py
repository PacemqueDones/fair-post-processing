import torch
from .pareto import pareto_front
from optimization.multiobjective import CommonDescent
from soft_metrics import SOFT_METRIC_MAP
from objectives.objectives import PerformancePreservationObjective

class FairPostProcessor:
    def __init__(self, model, objectives, selector, selection_metrics: dict | None =None, preserve_performance: bool = False, lr=1e-2, epochs=100):
        self.descent_ = CommonDescent()

        self.model = model
        self.objectives = objectives
        self.selector = selector
        self.selection_metrics = selection_metrics or {}
        self.preserve_performance = preserve_performance
        self.lr = lr
        self.epochs = epochs

        self.loss_history_ = []
        self.metric_history_ = []
        self.pareto_points_ = []
        self.pareto_front_ = []
        self.threshold_history_ = []
        self.best_thresholds_ = None

        self.metric_names_ = None
        self.metric_directions_ = None

    def _build_performance_preservation_objective(self, probs, y_true, sensitive_attr):
        perf_metrics = [
            metric for metric in self.selection_metrics
            if metric.type == "performance"
        ]

        soft_metrics = []
        for metric in perf_metrics:
            soft_cls = SOFT_METRIC_MAP.get(metric.name)
            if soft_cls is not None:
                soft_metrics.append(soft_cls())

        reference_values = {}
        for metric in soft_metrics:
            with torch.no_grad():
                reference_values[metric.name] = metric(
                    y_true=y_true,
                    scores=probs,
                    sensitive_attr=sensitive_attr
                ).detach()

        return PerformancePreservationObjective(
            differentiable_metrics=soft_metrics,
            reference_values=reference_values
        )

    def fit(self, probs, y_true, sensitive_attr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        probs = torch.tensor(probs, dtype=torch.float32)
        y_true = torch.tensor(y_true, dtype=torch.long)
        sensitive_attr = torch.tensor(sensitive_attr, dtype=torch.long)

        if self.preserve_performance:
            perf_obj = self._build_performance_preservation_objective(
                probs=probs,
                y_true=y_true,
                sensitive_attr=sensitive_attr
            )
            active_objectives = list(self.objectives) + [perf_obj]
        else:
            active_objectives = self.objectives

        for epoch in range(self.epochs):
            scores = self.model(probs)

            losses = []
            loss_dict = {}

            for obj in active_objectives:
                loss = obj(scores, y_true, sensitive_attr)
                losses.append(loss)
                loss_dict[obj.name] = loss.item()

            params = [p for p in self.model.parameters() if p.requires_grad]
            total_loss, alphas = self.descent_.combine(losses, params)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            self.loss_history_.append(loss_dict)

            with torch.no_grad():
                y_pred = torch.argmax(self.model(probs), dim=1)

                metric_dict = {}
                point = []

                for metric in self.selection_metrics:
                    value = metric(
                        y_true=y_true,
                        y_pred=y_pred,
                        sensitive_attr=sensitive_attr,
                        scores=scores
                    )

                    metric_dict[metric.name] = value
                    point.append(value)

                self.metric_history_.append(metric_dict)
                self.pareto_points_.append(point)
                self.threshold_history_.append(
                    self.model.thresholds.detach().clone()
                )

        self.metric_names_ = [metric.name for metric in self.selection_metrics]
        self.metric_directions_ = [metric.direction for metric in self.selection_metrics]

        front_idx = pareto_front(self.pareto_points_, self.metric_directions_)
        self.pareto_front_ = [self.pareto_points_[i] for i in front_idx]

        local_idx = self.selector.select(self.pareto_front_, self.metric_directions_)
        global_idx = front_idx[local_idx]

        self.best_thresholds_ = self.threshold_history_[global_idx]
        self.best_metrics_ = self.metric_history_[global_idx]
        self.best_losses_ = self.loss_history_[global_idx]

        return self

    def predict(self, probs):
        probs = torch.tensor(probs, dtype=torch.float32)
        with torch.no_grad():
            self.model.thresholds.copy_(self.best_thresholds_)
            scores = self.model(probs)
            return torch.argmax(scores, dim=1).cpu().numpy()

    def get_thresholds(self):
        return self.best_thresholds_
    

class LossFairPostProcessor:
    def __init__(self, model, objectives, selector, lr=1e-2, epochs=100):
        self.model = model
        self.objectives = objectives
        self.selector = selector
        self.lr = lr
        self.epochs = epochs

        self.pareto_points_ = []
        self.threshold_history_ = []
        self.best_thresholds_ = None
        self.metric_directions_ = None

    def fit(self, probs, y_true, sensitive_attr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        probs = torch.tensor(probs, dtype=torch.float32)
        y_true = torch.tensor(y_true, dtype=torch.long)
        sensitive_attr = torch.tensor(sensitive_attr, dtype=torch.long)

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            scores = self.model(probs)

            losses = []
            for obj in self.objectives:
                loss = obj(scores, y_true, sensitive_attr)
                losses.append(loss)

            total_loss = sum(losses)  # só para começar
            total_loss.backward()
            optimizer.step()

            point = [loss.item() for loss in losses]
            self.pareto_points_.append(point)
            self.threshold_history_.append(self.model.thresholds.detach().clone())

        idx = self.selector.select(self.pareto_points_)
        self.best_thresholds_ = self.threshold_history_[idx]
        return self

    def predict(self, probs):
        probs = torch.tensor(probs, dtype=torch.float32)
        with torch.no_grad():
            self.model.thresholds.copy_(self.best_thresholds_)
            scores = self.model(probs)
            return torch.argmax(scores, dim=1).cpu().numpy()

    def get_thresholds(self):
        return self.best_thresholds_