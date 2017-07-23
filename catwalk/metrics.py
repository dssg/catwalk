import numpy
from sklearn import metrics
from sklearn.metrics import confusion_matrix


"""Metric definitions

Mostly just wrappers around sklearn.metrics functions, these functions
implement a generalized interface to metric calculations that can be stored
as a scalar in the database.

All functions should take four parameters:
predictions_proba (1d array-like) Prediction probabilities
predictions_binary (1d array-like) Binarized predictions
labels (1d array-like) Ground truth target values
parameters (dict) Any needed hyperparameters in the implementation

All functions should be wrapped with @Metric to define the optimal direction

All functions should return: (float) the resulting score

Functions defined here are meant to be used in ModelEvaluator.available_metrics
"""

class Metric(object):
    """decorator for metrics: result will be a callable metric with an 
    `optimality` parameter defined as either 'minimize' or 'maximize' 
    depending on whether smaller or larger metric values indicate
    better models.
    """

    def __init__(self, optimality):
        if optimality not in ('minimize', 'maximize'):
            raise ValueError("optimality must be 'minimize' or 'maximize'")
        self.optimality = optimality

    def __call__(self, function, *params, **kwparams):

        class DecoratedMetric(object):
            def __init__(self, optimality, function):
                self.optimality = optimality
                self.function = function
                self.__name__ = function.__name__
                self.__doc__ = function.__doc__
            def __call__(self, *params, **kwparams):
                return self.function(*params, **kwparams)

        return DecoratedMetric(self.optimality, function)



@Metric('maximize')
def precision(_, predictions_binary, labels, parameters):
    return metrics.precision_score(labels, predictions_binary, **parameters)


@Metric('maximize')
def recall(_, predictions_binary, labels, parameters):
    return metrics.recall_score(labels, predictions_binary, **parameters)


@Metric('maximize')
def fbeta(_, predictions_binary, labels, parameters):
    return metrics.fbeta_score(labels, predictions_binary, **parameters)


@Metric('maximize')
def f1(_, predictions_binary, labels, parameters):
    return metrics.f1_score(labels, predictions_binary, **parameters)


@Metric('maximize')
def accuracy(_, predictions_binary, labels, parameters):
    return metrics.accuracy_score(labels, predictions_binary, **parameters)


@Metric('maximize')
def roc_auc(predictions_proba, _, labels, parameters):
    return metrics.roc_auc_score(labels, predictions_proba)


@Metric('maximize')
def avg_precision(predictions_proba, _, labels, parameters):
    return metrics.average_precision_score(labels, predictions_proba)


@Metric('maximize')
def true_positives(_, predictions_binary, labels, parameters):
    return int(confusion_matrix(labels, predictions_binary)[1,1])


@Metric('minimize')
def false_positives(_, predictions_binary, labels, parameters):
    return int(confusion_matrix(labels, predictions_binary)[0,1])


@Metric('maximize')
def true_negatives(_, predictions_binary, labels, parameters):
    return int(confusion_matrix(labels, predictions_binary)[0,0])


@Metric('minimize')
def false_negatives(_, predictions_binary, labels, parameters):
    return int(confusion_matrix(labels, predictions_binary)[1,0])


@Metric('minimize')
def fpr(_, predictions_binary, labels, parameters):
    fp = false_positives(_, predictions_binary, labels, parameters)
    return float(fp / labels.count(0))


class UnknownMetricError(ValueError):
    """Signifies that a metric name was passed, but no matching computation
    function is available
    """
    pass

