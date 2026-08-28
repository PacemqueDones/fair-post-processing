from __future__ import annotations

from torchjd.aggregation import (
    CAGrad,
    Mean,
    MGDA,
    PCGrad,
    UPGrad,
)


_AGGREGATORS = {
    "mean": Mean,
    "mgda": MGDA,
    "upgrad": UPGrad,
    "pcgrad": PCGrad,
    "cagrad": CAGrad,
}


def build_aggregator(name: str, **kwargs):
    """
    Constrói um agregador do TorchJD a partir de um nome simples.

    Exemplos
    --------
    build_aggregator("mgda")
    build_aggregator("upgrad")
    build_aggregator("pc-grad")
    """
    normalized_name = (
        name.strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
    )

    if normalized_name not in _AGGREGATORS:
        available = ", ".join(sorted(_AGGREGATORS))

        raise ValueError(
            f"Agregador desconhecido: {name!r}. "
            f"Opções disponíveis: {available}."
        )

    return _AGGREGATORS[normalized_name](**kwargs)
