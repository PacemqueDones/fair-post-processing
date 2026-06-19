from ..copsolver.frank_wolfe_solver import FrankWolfeSolver
from ..copsolver.analytical_solver import AnalyticalSolver

import torch
import numpy as np


def flatten_grads(grads, params):
    flat = []

    for g, p in zip(grads, params):
        if g is None:
            flat.append(torch.zeros_like(p).reshape(-1))
        else:
            flat.append(g.reshape(-1))

    return torch.cat(flat)


def safe_cosine(g1, g2, eps=1e-12):
    denom = torch.norm(g1) * torch.norm(g2)

    if denom < eps:
        return 0.0

    return float((torch.dot(g1, g2) / denom).detach().cpu())


class CommonDescent:
    def __init__(self, solver=None, normalize=True, eps=1e-8, min_scale_ratio=1e-3):
        self.solver = solver
        self.normalize = normalize

        self.last_alphas_ = None
        self.last_diagnostics_ = None

        self.initial_losses_ = None
        self.raw_initial_losses_ = None

        self.eps = eps
        self.min_scale_ratio = min_scale_ratio

    def _initialize_scales(self, losses):
        self.raw_initial_losses_ = [
            float(loss.detach().cpu())
            for loss in losses
        ]

        mean_scale = np.mean([
            abs(value)
            for value in self.raw_initial_losses_
        ])

        min_scale = max(
            self.eps,
            self.min_scale_ratio * mean_scale
        )

        self.initial_losses_ = [
            max(abs(value), min_scale)
            for value in self.raw_initial_losses_
        ]

    def _normalize_losses(self, losses):
        normalized_losses = []

        for i, loss in enumerate(losses):
            scale = self.initial_losses_[i]
            normalized_losses.append(loss / scale)

        return normalized_losses

    def _compute_flat_grads(self, losses, params):
        grads = []

        for loss in losses:
            g = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                allow_unused=True
            )

            grads.append(g)

        flat_grads = []

        for g in grads:
            fg = flatten_grads(g, params)
            flat_grads.append(fg)

        return flat_grads

    def _build_diagnostics(
        self,
        losses,
        normalized_losses,
        raw_grads,
        normalized_grads,
        update_grads,
        alphas,
        total_loss
    ):
        objective_names = [
            f"obj_{i}"
            for i in range(len(losses))
        ]

        if len(losses) == 2:
            objective_names = [
                "cross_entropy",
                "laplacian_fairness"
            ]

        d = torch.zeros_like(update_grads[0])

        for alpha, g in zip(alphas, update_grads):
            d = d + float(alpha) * g

        losses_raw = {}
        losses_normalized = {}

        grad_norms_raw = {}
        grad_norms_normalized = {}

        descent_dot_raw = {}
        descent_dot_normalized = {}

        scales = {}

        for i, name in enumerate(objective_names):
            losses_raw[name] = float(losses[i].detach().cpu())
            losses_normalized[name] = float(normalized_losses[i].detach().cpu())

            grad_norms_raw[name] = float(torch.norm(raw_grads[i]).detach().cpu())
            grad_norms_normalized[name] = float(torch.norm(normalized_grads[i]).detach().cpu())

            descent_dot_raw[name] = float(torch.dot(raw_grads[i], d).detach().cpu())
            descent_dot_normalized[name] = float(torch.dot(normalized_grads[i], d).detach().cpu())

            scales[name] = float(self.initial_losses_[i])

        if len(losses) == 2:
            pair_name = f"{objective_names[0]}__{objective_names[1]}"

            cosine_raw = {
                pair_name: safe_cosine(raw_grads[0], raw_grads[1])
            }

            cosine_normalized = {
                pair_name: safe_cosine(normalized_grads[0], normalized_grads[1])
            }
        else:
            cosine_raw = {}
            cosine_normalized = {}

        return {
            "normalize": self.normalize,
            "scales": scales,

            "losses_raw": losses_raw,
            "losses_normalized": losses_normalized,

            "grad_norms_raw": grad_norms_raw,
            "grad_norms_normalized": grad_norms_normalized,

            "cosine_raw": cosine_raw,
            "cosine_normalized": cosine_normalized,

            "descent_dot_raw": descent_dot_raw,
            "descent_dot_normalized": descent_dot_normalized,

            "descent_norm": float(torch.norm(d).detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),

            "alphas": [
                float(alpha)
                for alpha in alphas
            ],

            # compatibilidade com seus diagnósticos antigos
            "grad_norms": grad_norms_normalized,
            "cosine_similarity": cosine_normalized,
        }

    def combine(self, losses, params):
        if self.initial_losses_ is None:
            self._initialize_scales(losses)

        normalized_losses = self._normalize_losses(losses)

        raw_grads = self._compute_flat_grads(losses, params)
        normalized_grads = self._compute_flat_grads(normalized_losses, params)

        if self.normalize:
            solver_grads = normalized_grads
            update_losses = normalized_losses
        else:
            solver_grads = raw_grads
            update_losses = losses

        flat_grads_np = [
            g.detach().cpu().numpy()
            for g in solver_grads
        ]

        solver = self.solver

        if solver is None:
            solver = AnalyticalSolver() if len(losses) == 2 else FrankWolfeSolver()

        alphas = solver.solve(flat_grads_np)
        self.last_alphas_ = alphas

        total_loss = sum(
            float(alpha) * loss
            for alpha, loss in zip(alphas, update_losses)
        )

        self.last_diagnostics_ = self._build_diagnostics(
            losses=losses,
            normalized_losses=normalized_losses,
            raw_grads=raw_grads,
            normalized_grads=normalized_grads,
            update_grads=solver_grads,
            alphas=alphas,
            total_loss=total_loss
        )

        return total_loss, alphas