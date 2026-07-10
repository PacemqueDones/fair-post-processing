import torch
import torch.nn.functional as F
from .objective import Objective

class LaplacianFairnessObjective(Objective):
    name = "laplacian_fairness"

    def __init__(
        self,
        L,
        W=None,
        fairness_weight=1.0,
        ce_weight=0,
        normalize="samples",
        symmetrize=False,
        eps=1e-8,
    ):
        valid_normalizations = {None, "samples", "edges"}

        if normalize not in valid_normalizations:
            raise ValueError("normalize must be None, 'samples' or 'edges'.")

        if normalize == "edges" and W is None:
            raise ValueError("W must be provided when normalize='edges'.")

        L = L.float()

        if symmetrize:
            L = 0.5 * (L + L.T)

        self.L = L
        self.W = None if W is None else W.float()

        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight
        self.normalize = normalize
        self.symmetrize = symmetrize
        self.eps = eps

    def __call__(self, logits, y_true, sensitive_attr):
        F_scores = logits.float()

        if F_scores.ndim == 1:
            F_scores = F_scores.view(-1, 1)

        L = self.L.to(device=F_scores.device, dtype=F_scores.dtype,)

        if L.shape[0] != F_scores.shape[0]:
            raise ValueError(
                f"L tem shape {L.shape}, mas logits têm shape "
                f"{F_scores.shape}. O Laplaciano precisa ser "
                "gerado sobre o mesmo conjunto usado no fit."
            )

        # Forma densa compatível com TorchJD:
        # tr(F^T L F) = sum(F * (L F))
        LF = L @ F_scores
        fairness = torch.sum(F_scores * LF)

        if self.normalize == "samples":
            fairness = fairness / F_scores.shape[0]

        elif self.normalize == "edges":
            W = self.W.to(device=F_scores.device, dtype=F_scores.dtype,)

            if W.shape != L.shape:
                raise ValueError(f"W tem shape {W.shape}, mas L tem shape {L.shape}.")

            edge_weight_sum = W.sum()

            fairness = (2.0 * fairness / edge_weight_sum.clamp_min(self.eps))

        if self.ce_weight > 0:
            ce = F.cross_entropy(logits, y_true)
        else:
            ce = 0

        loss = (self.fairness_weight * fairness + self.ce_weight * ce)

        return [loss], [self.name]
    

class NearestNeighborLaplacianFairnessObjective(Objective):
    """Regularização laplaciana calculada pelas arestas.

    Para um grafo simétrico cujas arestas não direcionadas
    são armazenadas uma única vez, calcula:

        sum_{ {i,j} in E }
            w_ij ||F_i - F_j||_2^2

    Essa expressão é exatamente equivalente a:

        tr(F^T L F)

    para o Laplaciano não normalizado:

        L = D - W.
    """

    name = "laplacian_fairness"

    def __init__(
        self,
        edge_index,
        edge_weight,
        fairness_weight=1.0,
        ce_weight=0.0,
        normalize="samples",
        eps=1e-8,
    ):
        valid_normalizations = {
            None,
            "samples",
            "edges",
        }

        if normalize not in valid_normalizations:
            raise ValueError(
                "normalize deve ser None, "
                "'samples' ou 'edges'."
            )

        edge_index = torch.as_tensor(edge_index, dtype=torch.long)

        edge_weight = torch.as_tensor(edge_weight, dtype=torch.float32).reshape(-1)

        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
        ):
            raise ValueError(
                "edge_index deve possuir shape "
                "(2, n_edges)."
            )

        if edge_index.shape[1] != edge_weight.shape[0]:
            raise ValueError(
                "A quantidade de arestas em edge_index "
                "deve ser igual à quantidade de pesos "
                "em edge_weight."
            )

        if edge_index.numel() > 0:
            if edge_index.min() < 0:
                raise ValueError(
                    "edge_index não pode possuir "
                    "índices negativos."
                )

        if torch.any(edge_weight < 0):
            raise ValueError(
                "Os pesos das arestas devem ser "
                "não negativos."
            )

        self.edge_index = edge_index
        self.edge_weight = edge_weight

        self.fairness_weight = fairness_weight
        self.ce_weight = ce_weight
        self.normalize = normalize
        self.eps = eps

    def __call__(
        self,
        logits,
        y_true,
        sensitive_attr,
    ):
        # Mantém a decisão do seu objetivo anterior:
        # a regularização atua diretamente nos logits.
        F_scores = torch.softmax(logits.float(),dim=1)

        if F_scores.ndim == 1:
            F_scores = F_scores.view(-1, 1)

        edge_index = self.edge_index.to(
            device=F_scores.device,
        )

        edge_weight = self.edge_weight.to(
            device=F_scores.device,
            dtype=F_scores.dtype,
        )

        if edge_index.numel() == 0:
            fairness = F_scores.sum() * 0.0

        else:
            max_index = edge_index.max().item()

            if max_index >= F_scores.shape[0]:
                raise ValueError(
                    "edge_index contém um índice maior "
                    "que o número de amostras dos logits. "
                    f"Maior índice: {max_index}; "
                    f"número de amostras: "
                    f"{F_scores.shape[0]}."
                )

            source = edge_index[0]
            target = edge_index[1]

            source_scores = F_scores[source]
            target_scores = F_scores[target]

            squared_difference = (
                source_scores - target_scores
            ).pow(2).sum(dim=1)

            fairness = torch.sum(
                edge_weight * squared_difference
            )

            if self.normalize == "samples":
                fairness = (
                    fairness
                    / F_scores.shape[0]
                )

            elif self.normalize == "edges":
                fairness = (
                    fairness
                    / edge_weight.sum().clamp_min(
                        self.eps
                    )
                )

        if self.ce_weight > 0:
            ce = F.cross_entropy(
                logits,
                y_true,
            )
        else:
            ce = logits.sum() * 0.0

        loss = (
            self.fairness_weight * fairness
            + self.ce_weight * ce
        )

        return [loss], [self.name]