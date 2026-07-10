from dataclasses import dataclass

import torch


@dataclass
class FairPairGeometry:
    """Geometria baseada em pares de indivíduos.

    pair_index:
        Tensor de shape (2, M).

        Cada coluna representa um par:

            pair_index[:, r] = [i_r, j_r]

    raw_distances:
        Distâncias geométricas originais dos pares.

    fair_distances:
        Distâncias normalizadas usadas pelas métricas IFV.

    edge_weights:
        Pesos usados pela regularização laplaciana.

        Pode ser None quando a geometria será usada
        somente para métricas.

    num_samples:
        Número de indivíduos do conjunto.

    num_total_pairs:
        Número total de pares possíveis:

            n * (n - 1) // 2

    num_stored_pairs:
        Número de pares efetivamente armazenados.

    is_exact:
        True quando todos os pares foram armazenados.
    """

    pair_index: torch.Tensor
    raw_distances: torch.Tensor
    fair_distances: torch.Tensor

    num_samples: int
    num_total_pairs: int
    num_stored_pairs: int
    is_exact: bool

    edge_weights: torch.Tensor | None = None
    tau: float | None = None