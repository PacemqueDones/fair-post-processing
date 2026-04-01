import torch
from .pareto import pareto_front
from fairpp.optimization.multiobjective import CommonDescent
from .soft_metrics import SOFT_METRIC_MAP
from .objectives.objectives import PerformancePreservationObjective
from .filter.performance_filter import PerformanceRangeFilter

class FairPostProcessor:
    def __init__(self, model, objectives, selector, selection_metrics: list | None = None, preserve_performance: bool = False, performance_tolerance=0.1, lr=1e-2, epochs=100):
        self.descent_ = CommonDescent()

        self.model = model
        self.objectives = objectives
        self.selector = selector
        self.selection_metrics = selection_metrics or []
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
        self.metric_type_ = None
        self.performance_reference_metrics_ = None

        self.preserve_performance = preserve_performance
        self.performance_tolerance = performance_tolerance
        self.feasible_filter  = None
        self.feasible_pareto_front_ = None
        self.fallback_used_ = False
        self.fallback_reason_ = None
        self.filter_info_ = None
        
        self.alpha_history_ = []
        self.total_loss_history_ = []

        self.grad_norm_history_ = []
        self.cosine_similarity_history_ = []
        self.loss_delta_history_ = []
        self.step_norm_history_ = []
        
    def _build_performance_reference_metrics(self, probs, y_true, sensitive_attr):
        with torch.no_grad():
            scores = probs
            y_pred = torch.argmax(scores, dim=1)

            ref = {}
            for metric in self.selection_metrics:
                if metric.type != "performance":
                    continue

                ref[metric.name] = metric(
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_attr=sensitive_attr,
                    scores=scores
                )

        return ref

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

        self.performance_reference_metrics_ = self._build_performance_reference_metrics(probs, y_true, sensitive_attr)

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

            '''params = [p for p in self.model.parameters() if p.requires_grad]

            grad_norms = {}
            grads_per_obj = {}

            for obj, loss in zip(active_objectives, losses):
                grads = torch.autograd.grad(
                    loss,
                    params,
                    retain_graph=True,
                    allow_unused=True
                )

                flat = []
                for g in grads:
                    if g is None:
                        continue
                    flat.append(g.reshape(-1))

                if flat:
                    gvec = torch.cat(flat)
                    grad_norms[obj.name] = float(torch.norm(gvec).detach().cpu())
                    grads_per_obj[obj.name] = gvec.detach()
                else:
                    grad_norms[obj.name] = 0.0
                    grads_per_obj[obj.name] = None

            if (
                grads_per_obj["cross_entropy"] is not None
                and grads_per_obj["demographic_parity"] is not None
            ):
                g1 = grads_per_obj["cross_entropy"]
                g2 = grads_per_obj["demographic_parity"]

                cos = torch.dot(g1, g2) / (torch.norm(g1) * torch.norm(g2) + 1e-12)
                cosine_similarity = float(cos.detach().cpu())
            else:
                cosine_similarity = None'''

            params = [p for p in self.model.parameters() if p.requires_grad]
            total_loss, alphas = self.descent_.combine(losses, params)

            self.total_loss_history_.append(float(total_loss.detach().cpu()))
            self.alpha_history_.append(
                alphas.detach().cpu().tolist() if torch.is_tensor(alphas) else list(alphas)
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            self.loss_history_.append(loss_dict)

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
                        scores=scores_eval
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
        self.metric_type_ = [metric.type for metric in self.selection_metrics]

        front_idx = pareto_front(self.pareto_points_, self.metric_directions_)
        self.pareto_front_ = [self.pareto_points_[i] for i in front_idx]
        pareto_metric_history = [self.metric_history_[i] for i in front_idx]

        if self.preserve_performance:
            self.feasible_filter = PerformanceRangeFilter(self.performance_tolerance)
            feasible_idx, filter_info  = self.feasible_filter.get_indices(pareto_metric_history, self.selection_metrics, self.performance_reference_metrics_)
            self.feasible_pareto_front_ = [self.pareto_front_[i] for i in feasible_idx]
            local_idx = self.selector.select(self.feasible_pareto_front_, self.metric_directions_)
            selected_pareto_idx = feasible_idx[local_idx]
            global_idx = front_idx[selected_pareto_idx]

            self.filter_info_ = filter_info
            self.fallback_used_ = filter_info["fallback_used"]
            self.fallback_reason_ = filter_info["fallback_reason"]
        
        else:
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