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

        self.performance_reference_metrics_ = self._build_performance_reference_metrics(
            probs, y_true, sensitive_attr
        )

        if self.preserve_performance:
            perf_obj = self._build_performance_preservation_objective(
                probs=probs,
                y_true=y_true,
                sensitive_attr=sensitive_attr
            )
            active_objectives = list(self.objectives) + [perf_obj]
        else:
            active_objectives = list(self.objectives)

        params = [p for p in self.model.parameters() if p.requires_grad]

        for epoch in range(self.epochs):
            # -------------------------------------------------
            # Forward
            # -------------------------------------------------
            scores = self.model(probs)

            losses = []
            loss_dict = {}

            for obj in active_objectives:
                loss = obj(scores, y_true, sensitive_attr)
                losses.append(loss)
                loss_dict[obj.name] = float(loss.detach().cpu())

            # -------------------------------------------------
            # Gradientes individuais para diagnóstico
            # -------------------------------------------------
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

            self.grad_norm_history_.append(grad_norms)

            # -------------------------------------------------
            # Similaridade cosseno entre gradientes
            # -------------------------------------------------
            cosine_dict = {}

            if len(active_objectives) == 2:
                name1 = active_objectives[0].name
                name2 = active_objectives[1].name

                g1 = grads_per_obj.get(name1)
                g2 = grads_per_obj.get(name2)

                if g1 is not None and g2 is not None:
                    cos = torch.dot(g1, g2) / (torch.norm(g1) * torch.norm(g2) + 1e-12)
                    cosine_dict[f"{name1}__{name2}"] = float(cos.detach().cpu())
                else:
                    cosine_dict[f"{name1}__{name2}"] = None

            else:
                # opcional: guardar None ou dicionário vazio quando há mais de 2 objetivos
                cosine_dict = {}

            self.cosine_similarity_history_.append(cosine_dict)

            # -------------------------------------------------
            # Combinação multiobjetivo
            # -------------------------------------------------
            total_loss, alphas = self.descent_.combine(losses, params)

            self.total_loss_history_.append(float(total_loss.detach().cpu()))
            self.alpha_history_.append(
                alphas.detach().cpu().tolist() if torch.is_tensor(alphas) else list(alphas)
            )

            # -------------------------------------------------
            # Norma do passo dos thresholds
            # -------------------------------------------------
            old_thresholds = self.model.thresholds.detach().clone()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            new_thresholds = self.model.thresholds.detach().clone()
            step_norm = torch.norm(new_thresholds - old_thresholds).item()
            self.step_norm_history_.append(step_norm)

            # -------------------------------------------------
            # Histórico das losses e delta entre épocas
            # -------------------------------------------------
            self.loss_history_.append(loss_dict)

            if len(self.loss_history_) == 1:
                self.loss_delta_history_.append({k: 0.0 for k in loss_dict})
            else:
                prev = self.loss_history_[-2]
                curr = self.loss_history_[-1]
                delta = {k: curr[k] - prev[k] for k in curr}
                self.loss_delta_history_.append(delta)

            # -------------------------------------------------
            # Avaliação consistente após update
            # -------------------------------------------------
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

                    # força float simples para evitar problemas depois
                    metric_value = float(value)
                    metric_dict[metric.name] = metric_value
                    point.append(metric_value)

                self.metric_history_.append(metric_dict)
                self.pareto_points_.append(point)
                self.threshold_history_.append(
                    self.model.thresholds.detach().clone()
                )

        # -------------------------------------------------
        # Metadados das métricas
        # -------------------------------------------------
        self.metric_names_ = [metric.name for metric in self.selection_metrics]
        self.metric_directions_ = [metric.direction for metric in self.selection_metrics]
        self.metric_type_ = [metric.type for metric in self.selection_metrics]

        # -------------------------------------------------
        # Fronteira de Pareto
        # -------------------------------------------------
        front_idx = pareto_front(self.pareto_points_, self.metric_directions_)
        self.pareto_front_ = [self.pareto_points_[i] for i in front_idx]
        pareto_metric_history = [self.metric_history_[i] for i in front_idx]

        # -------------------------------------------------
        # Filtro de performance + seleção final
        # -------------------------------------------------
        if self.preserve_performance:
            self.feasible_filter = PerformanceRangeFilter(self.performance_tolerance)

            feasible_idx, filter_info = self.feasible_filter.get_indices(
                pareto_metric_history,
                self.selection_metrics,
                self.performance_reference_metrics_
            )

            self.feasible_pareto_front_ = [self.pareto_front_[i] for i in feasible_idx]

            local_idx = self.selector.select(
                self.feasible_pareto_front_,
                self.metric_directions_
            )

            selected_pareto_idx = feasible_idx[local_idx]
            global_idx = front_idx[selected_pareto_idx]

            self.filter_info_ = filter_info
            self.fallback_used_ = filter_info["fallback_used"]
            self.fallback_reason_ = filter_info["fallback_reason"]

        else:
            local_idx = self.selector.select(self.pareto_front_, self.metric_directions_)
            global_idx = front_idx[local_idx]

        # -------------------------------------------------
        # Melhor solução final
        # -------------------------------------------------
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