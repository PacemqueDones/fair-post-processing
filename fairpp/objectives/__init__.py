from .objective import Objective

from .performance import (
    CrossEntropyObjective,
)

from .group import (
    DemographicParityObjective,
    EqualityOpportunityObjective,
    WassersteinEqualityOpportunityObjective,
    WassersteinEqualityOpportunityQuantileObjective,
)

from .individual import (
    LaplacianFairnessObjective,
    SampledLaplacianFairnessObjective,
)


__all__ = [
    "Objective",

    # Performance
    "CrossEntropyObjective",

    # Group fairness
    "DemographicParityObjective",
    "EqualityOpportunityObjective",
    "WassersteinEqualityOpportunityObjective",
    "WassersteinEqualityOpportunityQuantileObjective",

    # Individual fairness
    "LaplacianFairnessObjective",
    "SampledLaplacianFairnessObjective",
]