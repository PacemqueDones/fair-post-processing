import torch
from .pareto import pareto_front
from optimization.multiobjective import CommonDescent
from gradients.gradients import GradientDiagnostics


class FairPostProcessor:
    def __init__(self, model, objectives, selector, selection_metrics: list | None = None, lr=1e-2, epochs=100, track_gradients: bool = False):
        self.model = model
        self.objectives = objectives
        self.selector = selector
        self.selection_metrics = selection_metrics or {}
        self.lr = lr
        self.epochs = epochs

        self.descent_ = CommonDescent()
        self.gradient_diagnostics = GradientDiagnostics(self.objectives) if track_gradients else None
        self.track_gradients = track_gradients

        self.loss_history_ = []
        self.metric_history_ = []
        self.pareto_points_ = []
        self.pareto_front_ = []
        self.threshold_history_ = []
        self.best_thresholds_ = None

        self.metric_names_ = None
        self.metric_directions_ = None

        self.cosine_similarity_history_ = []
        self.grad_norm_history_ = []
        self.total_loss_history_ = []
        self.alpha_history_ = []

    def _compute_losses(self, scores, y_true, sensitive_attr):
        losses = []
        loss_dict = {}

        for obj in self.objectives:
            loss = obj(scores, y_true, sensitive_attr)
            losses.append(loss)
            loss_dict[obj.name] = loss.item()

        return losses, loss_dict
    
    def _get_trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def _record_diagnostics(self, grad_norms, cosine_dict, total_loss, alphas):
        self.grad_norm_history_.append(grad_norms)
        self.cosine_similarity_history_.append(cosine_dict)
        self.total_loss_history_.append(float(total_loss.detach().cpu()))
        self.alpha_history_.append(
            alphas.detach().cpu().tolist() if torch.is_tensor(alphas) else list(alphas)
        )

    def fit(self, probs, y_true, sensitive_attr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        probs = torch.tensor(probs, dtype=torch.float32)
        y_true = torch.tensor(y_true, dtype=torch.long)
        sensitive_attr = torch.tensor(sensitive_attr, dtype=torch.long)

        for epoch in range(self.epochs):
            scores = self.model(probs)

            losses, loss_dict = self._compute_losses(scores, y_true, sensitive_attr)
            params = self._get_trainable_params()

            if self.track_gradients:
                grad_norms, cosine_dict = self.gradient_diagnostics.collect(losses, params)
            else:
                grad_norms, cosine_dict = {}, {}

            total_loss, alphas = self.descent_.combine(losses, params)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            self.loss_history_.append(loss_dict)

            if self.track_gradients:
                self._record_diagnostics(grad_norms, cosine_dict, total_loss, alphas)

            with torch.no_grad():
                scores_eval = self.model(probs)
                y_pred = torch.argmax(scores_eval, dim=1)

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
        
    def predict_proba(self, probs):
        probs = torch.tensor(probs, dtype=torch.float32)
        with torch.no_grad():
            self.model.thresholds.copy_(self.best_thresholds_)
            scores = self.model(probs)
            return torch.softmax(scores, dim=1).cpu().numpy()

    def get_thresholds(self):
        return self.best_thresholds_