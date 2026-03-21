import torch
from .pareto import pareto_front

class FairPostProcessor:
    def __init__(self, model, objectives, selector, selection_metrics: dict | None =None, lr=1e-2, epochs=100):
        self.model = model
        self.objectives = objectives
        self.selector = selector
        self.selection_metrics = selection_metrics or {}
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

    def fit(self, probs, y_true, sensitive_attr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        probs = torch.tensor(probs, dtype=torch.float32)
        y_true = torch.tensor(y_true, dtype=torch.long)
        sensitive_attr = torch.tensor(sensitive_attr, dtype=torch.long)

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            scores = self.model(probs)

            losses = []
            loss_dict = {}

            for obj in self.objectives:
                loss = obj(scores, y_true, sensitive_attr)
                losses.append(loss)
                loss_dict[obj.name] = loss.item()

            total_loss = sum(losses)
            total_loss.backward()
            optimizer.step()

            self.loss_history_.append(loss_dict)

            with torch.no_grad():
                y_pred = torch.argmax(self.model(probs), dim=1)

                metric_dict = {}
                point = []

                for name, metric in self.selection_metrics.items():
                    value = metric(
                        y_true=y_true,
                        y_pred=y_pred,
                        sensitive_attr=sensitive_attr,
                        scores=scores
                    )
                    metric_dict[name] = value
                    point.append(value)

                self.metric_history_.append(metric_dict)
                self.pareto_points_.append(point)
                self.threshold_history_.append(
                    self.model.thresholds.detach().clone()
                )

        self.metric_names_ = [metric.name for metric in self.selection_metrics.values()]
        self.metric_directions_ = [metric.direction for metric in self.selection_metrics.values()]

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