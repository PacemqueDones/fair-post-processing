from __future__ import annotations

from typing import Sequence

import torch
from torchjd.autojac import backward as jacobian_backward
from torchjd.autojac import jac_to_grad

from .aggregators import build_aggregator


def _flatten_grads(
    grads: Sequence[torch.Tensor | None],
    params: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    """
    Concatena os gradientes de todos os parâmetros.

    Quando um objetivo não depende de algum parâmetro, insere zeros
    no trecho correspondente para manter todas as linhas da Jacobiana
    com o mesmo tamanho.
    """
    pieces = []

    for grad, param in zip(grads, params):
        if grad is None:
            pieces.append(torch.zeros_like(param).reshape(-1))
        else:
            pieces.append(grad.reshape(-1))

    return torch.cat(pieces)


def _safe_cosine_matrix(
    jacobian: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    gramian = jacobian @ jacobian.T
    norms = torch.linalg.vector_norm(jacobian, dim=1)

    denominator = (
        norms[:, None] * norms[None, :]
    ).clamp_min(eps)

    return gramian / denominator


class JacobianDescent:
    """
    Integra o TorchJD ao FairPostProcessor.

    O TorchJD é responsável pela direção efetivamente usada pelo
    otimizador. Esta classe também calcula, antes da chamada ao
    TorchJD, os diagnósticos necessários para o histórico do projeto.
    """

    def __init__(
        self,
        aggregator: str = "upgrad",
        normalize_objectives: bool = False,
        aggregator_kwargs: dict | None = None,
        eps: float = 1e-8,
        min_scale_ratio: float = 1e-3,
    ):
        self.aggregator_name = aggregator
        self.normalize_objectives = normalize_objectives
        self.eps = eps
        self.min_scale_ratio = min_scale_ratio

        self.aggregator = build_aggregator(
            aggregator,
            **(aggregator_kwargs or {}),
        )

        self.initial_losses_: list[float] | None = None
        self.raw_initial_losses_: list[float] | None = None

        self.last_jacobian_ = None
        self.last_direction_ = None
        self.last_weights_ = None
        self.last_diagnostics_ = None

    def _initialize_scales(
        self,
        losses: Sequence[torch.Tensor],
    ) -> None:
        self.raw_initial_losses_ = [
            float(loss.detach().cpu())
            for loss in losses
        ]

        absolute = [
            abs(value)
            for value in self.raw_initial_losses_
        ]

        mean_scale = sum(absolute) / max(len(absolute), 1)

        min_scale = max(
            self.eps,
            self.min_scale_ratio * mean_scale,
        )

        self.initial_losses_ = [
            max(value, min_scale)
            for value in absolute
        ]

    def _prepare_losses(
        self,
        losses: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        if self.initial_losses_ is None:
            self._initialize_scales(losses)

        if not self.normalize_objectives:
            return list(losses)

        return [
            loss / scale
            for loss, scale in zip(losses, self.initial_losses_)
        ]

    def _compute_jacobian(
        self,
        losses: Sequence[torch.Tensor],
        params: Sequence[torch.nn.Parameter],
    ) -> torch.Tensor:
        rows = []

        for loss in losses:
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

            rows.append(
                _flatten_grads(grads, params)
            )

        return torch.stack(rows)

    @staticmethod
    def _flatten_parameter_grads(
        params: Sequence[torch.nn.Parameter],
    ) -> torch.Tensor:
        pieces = []

        for param in params:
            if param.grad is None:
                pieces.append(
                    torch.zeros_like(param).reshape(-1)
                )
            else:
                pieces.append(
                    param.grad.reshape(-1)
                )

        return torch.cat(pieces)

    def _recover_weights(
        self,
        jacobian: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recupera pesos w que satisfazem aproximadamente J^T w = direction.

        Isso é usado apenas para diagnóstico. A direção verdadeira continua
        sendo a produzida pelo agregador do TorchJD.
        """
        gramian = jacobian @ jacobian.T
        right_side = jacobian @ direction

        return torch.linalg.pinv(gramian) @ right_side

    def _build_diagnostics(
        self,
        raw_losses: Sequence[torch.Tensor],
        prepared_losses: Sequence[torch.Tensor],
        raw_jacobian: torch.Tensor,
        prepared_jacobian: torch.Tensor,
        direction: torch.Tensor,
        weights: torch.Tensor,
        objective_names: Sequence[str],
    ) -> dict:
        raw_norms = torch.linalg.vector_norm(
            raw_jacobian,
            dim=1,
        )

        prepared_norms = torch.linalg.vector_norm(
            prepared_jacobian,
            dim=1,
        )

        raw_cosine = _safe_cosine_matrix(
            raw_jacobian,
            self.eps,
        )

        prepared_cosine = _safe_cosine_matrix(
            prepared_jacobian,
            self.eps,
        )

        raw_dot = raw_jacobian @ direction
        prepared_dot = prepared_jacobian @ direction

        losses_raw = {
            name: float(loss.detach().cpu())
            for name, loss in zip(objective_names, raw_losses)
        }

        losses_normalized = {
            name: float(loss.detach().cpu())
            for name, loss in zip(objective_names, prepared_losses)
        }

        grad_norms_raw = {
            name: float(value.detach().cpu())
            for name, value in zip(objective_names, raw_norms)
        }

        grad_norms_normalized = {
            name: float(value.detach().cpu())
            for name, value in zip(objective_names, prepared_norms)
        }

        descent_dot_raw = {
            name: float(value.detach().cpu())
            for name, value in zip(objective_names, raw_dot)
        }

        descent_dot_normalized = {
            name: float(value.detach().cpu())
            for name, value in zip(objective_names, prepared_dot)
        }

        cosine_raw = {}
        cosine_normalized = {}

        for i in range(len(objective_names)):
            for j in range(i + 1, len(objective_names)):
                pair_name = (
                    f"{objective_names[i]}__"
                    f"{objective_names[j]}"
                )

                cosine_raw[pair_name] = float(
                    raw_cosine[i, j].detach().cpu()
                )

                cosine_normalized[pair_name] = float(
                    prepared_cosine[i, j].detach().cpu()
                )

        weights_cpu = weights.detach().cpu()
        prepared_vector = torch.stack([
            loss.detach()
            for loss in prepared_losses
        ])

        diagnostic_total_loss = torch.dot(
            weights.to(prepared_vector),
            prepared_vector,
        )

        return {
            "aggregator": self.aggregator_name,
            "normalize": self.normalize_objectives,

            "scales": {
                name: float(scale)
                for name, scale in zip(
                    objective_names,
                    self.initial_losses_,
                )
            },

            "losses_raw": losses_raw,
            "losses_normalized": losses_normalized,

            "grad_norms_raw": grad_norms_raw,
            "grad_norms_normalized": grad_norms_normalized,

            "cosine_raw": cosine_raw,
            "cosine_normalized": cosine_normalized,

            "descent_dot_raw": descent_dot_raw,
            "descent_dot_normalized": descent_dot_normalized,

            "descent_norm": float(
                torch.linalg.vector_norm(direction)
                .detach()
                .cpu()
            ),

            "total_loss": float(
                diagnostic_total_loss.detach().cpu()
            ),

            "weights": weights_cpu.tolist(),

            # Compatibilidade com os gráficos antigos.
            "alphas": weights_cpu.tolist(),
            "grad_norms": grad_norms_normalized,
            "cosine_similarity": cosine_normalized,
        }

    def backward(
        self,
        losses: Sequence[torch.Tensor],
        params: Sequence[torch.nn.Parameter],
        objective_names: Sequence[str] | None = None,
    ) -> None:
        if len(losses) == 0:
            raise ValueError(
                "É necessário fornecer pelo menos um objetivo."
            )

        params = list(params)

        if len(params) == 0:
            raise ValueError(
                "O modelo não possui parâmetros treináveis."
            )

        if objective_names is None:
            objective_names = [
                f"obj_{i}"
                for i in range(len(losses))
            ]

        if len(objective_names) != len(losses):
            raise ValueError(
                "objective_names deve ter o mesmo tamanho de losses."
            )

        prepared_losses = self._prepare_losses(losses)

        # Jacobianas calculadas apenas para diagnóstico.
        raw_jacobian = self._compute_jacobian(
            losses,
            params,
        )

        if self.normalize_objectives:
            prepared_jacobian = self._compute_jacobian(
                prepared_losses,
                params,
            )
        else:
            prepared_jacobian = raw_jacobian

        # Direção efetivamente produzida pelo TorchJD.
        loss_vector = torch.stack(prepared_losses)

        jacobian_backward(loss_vector)
        jac_to_grad(params, self.aggregator)

        direction = self._flatten_parameter_grads(params)

        # Recuperação dos pesos para diagnóstico e compatibilidade.
        weights = self._recover_weights(
            prepared_jacobian.detach(),
            direction.detach(),
        )

        self.last_jacobian_ = prepared_jacobian.detach().clone()
        self.last_direction_ = direction.detach().clone()
        self.last_weights_ = weights.detach().clone()

        self.last_diagnostics_ = self._build_diagnostics(
            raw_losses=losses,
            prepared_losses=prepared_losses,
            raw_jacobian=raw_jacobian.detach(),
            prepared_jacobian=prepared_jacobian.detach(),
            direction=direction.detach(),
            weights=weights.detach(),
            objective_names=objective_names,
        )
