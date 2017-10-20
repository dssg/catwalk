# coding: utf-8

import warnings

import numpy as np
from math import log, sqrt
import random

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES

MAX_INT = np.iinfo(np.int32).max

def flatten_list(l):
    """
    Simple utility to flatten a list down to one dimension even if the list
    contains elements of differing depth
    """
    res = []
    for i in l:
        if isinstance(i, list):
            res = res + flatten_list(i)
        else:
            res = res + [i]
    return res

DEPRECATION_MSG_1D = (
    "Passing 1d arrays as data is deprecated in 0.17 and will "
    "raise ValueError in 0.19. Reshape your data either using "
    "X.reshape(-1, 1) if your data has a single feature or "
    "X.reshape(1, -1) if it contains a single sample."
)

class CutOff(BaseEstimator, TransformerMixin):
    """
    Transforms features cutting values out of established range


    Args:
       feature_range: Range of allowed values, default=`(0,1)`

    Usage:
       The recommended way of using this is::

           from sklearn.pipeline import Pipeline

           minmax_scaler = preprocessing.MinMaxScaler()
           dsapp_cutoff = CutOff()
           lr  = linear_model.LogisticRegression()

           pipeline =Pipeline([
                 ('minmax_scaler',minmax_scaler),
                 ('dsapp_cutoff', dsapp_cutoff),
                 ('lr', lr)
           ])

           pipeline.fit(X_train, y_train)
           pipeline.predict(X_test)

    """
    def __init__(self, feature_range=(0,1), copy=True):
        self.feature_range = feature_range
        self.copy = copy

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        feature_range = self.feature_range

        X = check_array(X, copy=self.copy, ensure_2d=False, dtype=FLOAT_DTYPES)

        if X.ndim == 1:
            warnings.warn(DEPRECATION_MSG_1D, DeprecationWarning)


        if np.any(X > feature_range[1]) or np.any(X < feature_range[0]):
            warnings.warn(
                "You got data that are out of the range: {}"
                .format(feature_range)
            )


        X[X > feature_range[1]] = feature_range[1]
        X[X < feature_range[0]] = feature_range[0]

        return X


# feels pretty gross to have to specify the categorical columns in the constructor
# even before the object is aware of the data it's operating on, but doesn't seem
# like the fit method is flexible enough to specify it there if we're going to 
# use it in a pipeline. ugh.
class SubsetWithCategoricals(BaseEstimator, TransformerMixin):
    """
    Subsets features of an array treating categoricals as a group

    Args:
        max_features : int, float, string or None, optional (default=None)
            The number of features to subset down to:
                - If int, then subset to `max_features` features.
                - If float, then `max_features` is a percentage and
                  `int(max_features * n_features)` features are used.
                - If "auto", then `max_features=sqrt(n_features)`.
                - If "sqrt", then `max_features=sqrt(n_features)`.
                - If "log2", then `max_features=log2(n_features)`.
                - If None, then `max_features=n_features`.

        categoricals : list,
            List of groups of column indices to be considered associated
            with one another as categoricals. For instance [[1,2], [7,8,9]]
            would mean columns 1 & 2 are associated as one categorical and
            7, 8, and 9 are associated as a second one.

    Attributes:
        subset_indices : list,
            Indices of the chosen subset of columns in the original array.

        max_features_ : int,
            The inferred value of max_features.
    """
    def __init__(self, categoricals, max_features='sqrt', random_state=None, copy=True):
        self.max_features = max_features
        self.categoricals = categoricals
        self.random_state = random_state
        self.copy = copy

    def _infer_max_features(self, num_features):
        if isinstance(self.max_features, float):
            return int(self.max_features*num_features)
        elif isinstance(self.max_features, int):
            return self.max_features
        elif self.max_features in ['auto', 'sqrt']:
            return int(sqrt(num_features))
        elif self.max_features == 'log2':
            return int(log(num_features, 2))
        elif self.max_features is None:
            return num_features
        else:
            raise ValueError('Invalid value for max_features: %s' % self.max_features)

    def fit(self, X, y=None):
        if isinstance(self.random_state, int):
            random.seed(self.random_state)
        elif isinstance(self.random_state, np.random.RandomState):
            random.seed(self.random_state.randint(MAX_INT))

        features = list(range(X.shape[1]))

        all_cats = set(flatten_list(self.categoricals))
        non_cats = set(features) - all_cats

        # this will be a mixed list of column indices for non-categoricals
        # and lists of indices for categorics
        distinct_features = list(non_cats) + self.categoricals

        self.max_features_ = self._infer_max_features(len(distinct_features))
        if self.max_features_ > len(distinct_features):
            raise ValueError('Cannot subset to more than distinct features: %s vs %s' % (
                self.max_features_, len(distinct_features)))

        self.subset_indices = sorted(flatten_list(
            random.sample(distinct_features, self.max_features_)
        ))

        return self

    def transform(self, X):
        X = check_array(X, copy=self.copy, ensure_2d=False, dtype=FLOAT_DTYPES)
        return X[:, self.subset_indices]
