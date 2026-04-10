from fairpp.copsolver.frank_wolfe_solver import FrankWolfeSolver
from fairpp.copsolver.analytical_solver import AnalyticalSolver
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

class CommonDescent:
    def __init__(self, solver=None, eps=1e-12):
        self.solver = solver
        self.last_alphas_ = None
        self.initial_losses_ = None
        self.eps = eps

    def combine(self, losses, params):
        if self.initial_losses_ is None:
            self.initial_losses_ = [
                float(loss.detach().cpu()) for loss in losses
            ]

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
        for i, g in enumerate(grads):
            fg = flatten_grads(g, params)
            scale = max(self.initial_losses_[i], self.eps)
            fg = fg / scale
            flat_grads.append(fg.detach().cpu().numpy())

        solver = self.solver
        if solver is None:
            solver = AnalyticalSolver() if len(losses) == 2 else FrankWolfeSolver()

        alphas = solver.solve(flat_grads)
        self.last_alphas_ = alphas

        total_loss = sum(alpha * loss for alpha, loss in zip(alphas, losses))
        return total_loss, alphas