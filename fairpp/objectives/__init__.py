from .objective import Objective

from .performance import (
    CrossEntropyObjective,
)

from .group import (
    DemographicParityObjective,
    EqualityOpportunityObjective,
    WassersteinDemographicParityObjective,
    WassersteinEqualityOpportunityObjective,
)

from .individual import (
    LaplacianFairnessObjective,
    SampledLaplacianFairnessObjective,
)


__all__ = [
    "Objective",

    # Performance
    "CrossEntropyObjective",
    "KLPreservationObjective",
    "JensenShannonPreservationObjective"

    # Group fairness
    "DemographicParityObjective",
    "EqualityOpportunityObjective",
    "WassersteinDemographicParityObjective",
    "WassersteinEqualityOpportunityObjective",

    # Individual fairness
    "LaplacianFairnessObjective",
    "SampledLaplacianFairnessObjective",
]