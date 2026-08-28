# FairPP

**FairPP** is a Python library for fairness-aware post-processing of classification models.

The library modifies the predictions of an already trained classifier without requiring the original model to be retrained. FairPP formulates post-processing as a **multi-objective optimization problem**, allowing predictive performance and fairness objectives to be optimized simultaneously.

The framework is designed to support different post-processing models, fairness definitions, optimization strategies and solution-selection methods.

## Overview

A trained classifier produces probabilities

$$
p_i = f(x_i),
$$

where ($p_i$) is the probability vector predicted for sample ($i$).

FairPP applies a parameterized post-processing model

$$
z_i' = T_\theta(p_i, s_i),
$$

where:

- ($T_\theta$) is the post-processing transformation;
- ($\theta$) contains the trainable parameters of the post-processor;
- ($p_i$) is the prediction produced by the original classifier;
- ($s_i$) contains the sensitive attributes associated with sample ($i$);
- ($z_i'$) represents the transformed logits used to obtain the final prediction.

Instead of optimizing a single loss, FairPP can optimize several objectives simultaneously:

$$
\min_\theta
\left(
L_{\text{performance}}(\theta),
L_{\text{fairness},1}(\theta),
\ldots,
L_{\text{fairness},m}(\theta)
\right).
$$

This produces a set of candidate solutions representing different trade-offs between predictive performance and fairness.

The general workflow is therefore:

```text
Base classifier
      |
      v
Predicted probabilities
      |
      v
Post-processing model
      |
      v
Performance + Fairness objectives
      |
      v
Multi-objective optimization
      |
      v
Candidate solutions
      |
      v
Pareto front
      |
      v
Solution selection
      |
      v
Final post-processed classifier
```

## Installation

Clone the repository and install FairPP in editable mode:

```bash
pip install -e .
```

The main dependencies are:

- PyTorch
- TorchJD
- NumPy
- scikit-learn

## Basic usage

A FairPP experiment is composed of four main elements:

1. a post-processing model;
2. optimization objectives;
3. metrics used to evaluate candidate solutions;
4. a strategy for selecting a solution from the Pareto front.

For example:

```python
from fairpp.models import LogitCategoricalAdditiveModel

from fairpp.objectives import (
    CrossEntropyObjective,
    DemographicParityObjective,
)

from fairpp.metrics import (
    BalancedAccuracyMetric,
    DemographicParityMetric,
)

from fairpp.selection import TopsisSelector

from fairpp import FairPostProcessor
```

### 1. Define the post-processing model

```python
model = LogitCategoricalAdditiveModel(
    alpha=10,
    num_classes=2,
    category_sizes=[2],
)
```

`num_classes` defines the number of prediction classes.

`category_sizes` describes the number of categories of each sensitive attribute. For example,

```python
category_sizes=[2, 3]
```

represents two sensitive attributes: the first containing two categories and the second containing three.

### 2. Define the optimization objectives

```python
objectives = [
    CrossEntropyObjective(),

    DemographicParityObjective(
        fairness_weight=15,
        within_attribute_reduction="none",
        across_attribute_reduction="none",
    ),
]
```

In this example, predictive performance and demographic parity are optimized simultaneously.

FairPP treats these quantities as separate objectives rather than combining them into a single weighted scalar loss.

### 3. Create the post-processor

```python
postprocessor = FairPostProcessor(
    model=model,
    objectives=objectives,

    selector=TopsisSelector([1, 1]),

    selection_metrics=[
        BalancedAccuracyMetric(),

        DemographicParityMetric(
            within_attribute_reduction="max",
            across_attribute_reduction="max",
        ),
    ],

    aggregator="upgrad",
    lr=5e-3,
    epochs=2000,
)
```

The `aggregator` determines how gradients from the different objectives are combined during multi-objective optimization.

### 4. Fit

```python
postprocessor.fit(
    train_probs=probs_train,
    train_y_true=y_train,
    train_sensitive_attr=S_train,

    val_probs=probs_validation,
    val_y_true=y_validation,
    val_sensitive_attr=S_validation,
)
```

The original classifier is not retrained.

FairPP receives its predicted probabilities and trains only the post-processing transformation.

During training, candidate solutions are evaluated on the validation data and stored for posterior analysis.

### 5. Predict

```python
predictions = postprocessor.predict(
    probs_test,
    S_test,
)
```

or obtain transformed probabilities with:

```python
probabilities = postprocessor.predict_proba(
    probs_test,
    S_test,
)
```

## Pareto front

Because FairPP optimizes multiple objectives, there is generally no single solution that simultaneously minimizes every objective.

Instead, training generates solutions with different trade-offs.

FairPP identifies the non-dominated solutions and constructs a **Pareto front**.

Conceptually:

```text
Fairness
  ^
  |
  |  *
  |    *
  |       *
  |           *
  |                *
  +--------------------> Performance
```

A solution may preserve more predictive performance while achieving a smaller fairness improvement, whereas another may substantially improve fairness at the cost of a larger performance reduction.

The final solution can therefore be selected according to the requirements of the experiment.

## Solution selection

FairPP currently provides different strategies for selecting solutions from the Pareto front.

### TOPSIS

```python
from fairpp.selection import TopsisSelector

selector = TopsisSelector([1, 1])
```

TOPSIS selects a compromise solution considering the different evaluation metrics.

### Reference point

```python
from fairpp.selection import ReferenceSelector

selector = ReferenceSelector(
    targets={"ddp": target_ddp},
    mode="optimize",
)
```

Reference-based selection makes it possible to search the Pareto front according to a desired fairness or performance target.

This is useful for experiments such as:

> Among solutions that reduce demographic parity disparity by at least 50%, which one preserves the highest balanced accuracy?

## Sensitive attributes

FairPP supports multiple sensitive attributes.

The sensitive attribute matrix follows the structure

```text
(num_samples, num_sensitive_attributes)
```

For example:

```python
S.shape
# (10000, 2)
```

represents 10,000 samples evaluated according to two sensitive attributes.

Each sensitive attribute may also contain multiple categories.

This allows experiments involving structures such as:

```text
Sensitive attribute 1
├── category 0
└── category 1

Sensitive attribute 2
├── category 0
├── category 1
└── category 2
```

## Project structure

```text
fairpp/
├── diagnostics/
├── evaluation/
├── geometry/
├── metrics/
├── models/
├── objectives/
├── optimization/
├── selection/
└── postprocessor.py
```

### `models`

Parameterized transformations applied to the predictions produced by the base classifier.

### `objectives`

Differentiable objectives used during optimization.

These include predictive-performance objectives and fairness objectives.

### `metrics`

Metrics used to evaluate predictive performance, group fairness and individual fairness.

### `optimization`

Multi-objective gradient aggregation strategies used during training.

### `selection`

Pareto-front construction and strategies for selecting candidate solutions.

### `geometry`

Similarity, distance and graph structures used by individual-fairness objectives.

### `evaluation`

Utilities for evaluating post-processed predictions.

### `diagnostics`

Utilities for inspecting optimization behavior and trained post-processors.

## Experiments

Experimental scripts are kept outside the library:

```text
experiments/
```

The experiments are responsible for defining datasets, training protocols, validation procedures, trade-off analysis and result persistence.

This separation keeps the core `fairpp` package independent from a specific experimental protocol.

## Current version

**FairPP 4.0.0**

The current version represents the transition from the original threshold-based post-processing approach toward a more general multi-objective post-processing framework, including support for categorical transformations and multiple sensitive attributes.

FairPP is currently under active research development.