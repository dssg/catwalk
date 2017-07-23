from catwalk.metrics import fpr
from catwalk.evaluation import ModelEvaluator

def test_metric_directionality():
    """All metrics must be wrapped using the @Metric decorator available
    in catwalk.metrics to provide an `optimality` attribute which must
    be one of 'minimize' or 'maximize'.
    """
    for met in ModelEvaluator.available_metrics.values():
        assert hasattr(met, 'optimality')
        assert met.optimality in ('minimize', 'maximize')


def test_fpr():
    predictions_binary = \
        [1, 1, 1, 0, 0, 0, 0, 0]
    labels = \
        [1, 1, 0, 1, 0, 0, 0, 1]

    result = fpr([], predictions_binary, labels, [])
    # false positives = 1
    # total negatives = 4
    assert result == 0.25


