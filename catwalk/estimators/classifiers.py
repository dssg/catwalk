# coding: utf-8

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from catwalk.estimators.transformers import CutOff, SubsetWithCategoricals

import numpy as np
import random

MAX_INT = np.iinfo(np.int32).max

class ScaledLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    An in-place replacement for the scikit-learn's LogisticRegression.

    It incorporates the MaxMinScaler, and the CutOff as preparations
    for the  logistic regression.
    """
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):


        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs

        self.minmax_scaler = MinMaxScaler()
        self.dsapp_cutoff = CutOff()
        self.lr = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C,
                                     fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                                     random_state=random_state, solver=solver, max_iter=max_iter,
                                     multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

        self.pipeline =Pipeline([
            ('minmax_scaler', self.minmax_scaler),
            ('dsapp_cutoff', self.dsapp_cutoff),
            ('lr', self.lr)
        ])


    def fit(self, X, y = None):
        self.pipeline.fit(X, y)

        self.min_ = self.pipeline.named_steps['minmax_scaler'].min_
        self.scale_ = self.pipeline.named_steps['minmax_scaler'].scale_
        self.data_min_ = self.pipeline.named_steps['minmax_scaler'].data_min_
        self.data_max_ = self.pipeline.named_steps['minmax_scaler'].data_max_
        self.data_range_ = self.pipeline.named_steps['minmax_scaler'].data_range_

        self.coef_ = self.pipeline.named_steps['lr'].coef_
        self.intercept_ = self.pipeline.named_steps['lr'].intercept_

        self.classes_ = self.pipeline.named_steps['lr'].classes_

        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict_log_proba(self, X):
        return self.pipeline.predict_log_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)

    def score(self, X, y):
        return self.pipeline.score(X,y)


class CatInATreeClassifier(BaseEstimator, ClassifierMixin):
    """
    Fit a decision tree with a subset of features that respects categoricals

    Args:
        categoricals : list,
            List of groups of column indices to be considered associated
            with one another as categoricals. For instance [[1,2], [7,8,9]]
            would mean columns 1 & 2 are associated as one categorical and
            7, 8, and 9 are associated as a second one.
    """
    def __init__(self,
                 categoricals,
                 max_features='sqrt',
                 random_state=None,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_leaf_nodes=None,
                 min_impurity_split=1e-07,
                 class_weight=None,
                 presort=False):

        self.categoricals = categoricals
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight
        self.presort = presort

        self.subset_cols = SubsetWithCategoricals(
            categoricals=categoricals, max_features=max_features, random_state=random_state
        )
        self.tree = DecisionTreeClassifier(
            criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=1.0, random_state=random_state, max_leaf_nodes=max_leaf_nodes,
            min_impurity_split=min_impurity_split, class_weight=class_weight, presort=presort
        )

        self.pipeline = Pipeline([
            ('subset_cols', self.subset_cols),
            ('tree', self.tree)
        ])

    def fit(self, X, y):

        # set the underlying random states before fitting
        # doing this here rather than in the constructor because self.random_state might
        # have been modified by an ensemble method
        self.pipeline.named_steps['subset_cols'].set_params(random_state=self.random_state)
        self.pipeline.named_steps['tree'].set_params(random_state=self.random_state)

        self.pipeline.fit(X, y)

        self.max_features_ = self.pipeline.named_steps['subset_cols'].max_features_
        self.subset_indices = self.pipeline.named_steps['subset_cols'].subset_indices

        self.classes_ = self.pipeline.named_steps['tree'].classes_
        self.n_classes_ = self.pipeline.named_steps['tree'].n_classes_
        self.n_features_ = self.pipeline.named_steps['tree'].n_features_
        self.n_outputs_ = self.pipeline.named_steps['tree'].n_outputs_
        self.tree_ = self.pipeline.named_steps['tree'].tree_

        # feature importances need to reference full column set but underlying tree
        # was trained on the subset, so fill in others with zeros
        fi = self.pipeline.named_steps['tree'].feature_importances_
        fi_dict = dict(zip(self.subset_indices, fi))
        fi_full = []
        for i in range(X.shape[1]):
            fi_full.append(fi_dict.get(i, 0))
        self.feature_importances_ = fi_full

        return self

    def apply(self, X):
        return self.pipeline.apply(X)

    def decision_path(self, X):
        return self.pipeline.decision_path(X)

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_log_proba(self, X):
        return self.pipeline.predict_log_proba(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def score(self, X, y):
        return self.pipeline.score(X, y)


class CatInAForestClassifier(BaggingClassifier):
    """
    Bagged classifier using CatInATreeClassifiers as estimators.
    Note that max_features is required here for the underlying
    subsetting and that the bagging classifier will use all selected
    features for each tree with no option for feature bootstrapping.
    """
    def __init__(self, categoricals, max_features_tree='sqrt', random_state=None,
        n_estimators=10, max_samples=1.0, bootstrap=True, oob_score=False, 
        warm_start=False, n_jobs=1, verbose=0, criterion="gini", splitter="best", 
        max_depth=None, min_samples_split=2, min_samples_leaf=1, 
        min_weight_fraction_leaf=0., max_leaf_nodes=None, min_impurity_split=1e-07, 
        class_weight=None, presort=False):

        # if isinstance(random_state, int):
        #     random.seed(random_state)
        # elif isinstance(random_state, np.random.RandomState):
        #     random.seed(random_state.randint(MAX_INT))

        # set up the base estimator as a CatInATreeClassifier()
        self.base_estimator = CatInATreeClassifier(
            categoricals=categoricals, max_features=max_features_tree, criterion=criterion, 
            splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, 
            max_leaf_nodes=max_leaf_nodes, min_impurity_split=min_impurity_split, 
            class_weight=class_weight, presort=presort
            )

        # Call the super-class's constructor
        # Here, we force each tree to consider all features (without bootstrapping)
        # as we'll handle the subsetting in the base estimator to have control over
        # sampling categoricals. Also note that calling the BaggingClassifier
        # constructor will set an object parameter `max_features`=1.0, so we've
        # nammed the class parameter `max_features_tree` avoid collision.
        BaggingClassifier.__init__(
            self, 
            base_estimator=self.base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=1.0,
            bootstrap=bootstrap,
            bootstrap_features=False,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
            )

        self.categoricals = categoricals
        self.max_features_tree = max_features_tree
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight
        self.presort = presort
