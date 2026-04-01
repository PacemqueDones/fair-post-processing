from .soft_metrics import SoftAccuracyMetric, SoftPrecisionMetric, SoftRecallMetric, SoftF1ScoreMetric

SOFT_METRIC_MAP = {
    "acc": SoftAccuracyMetric,
    "precision": SoftPrecisionMetric,
    "rec": SoftRecallMetric,
    "f1": SoftF1ScoreMetric,
}