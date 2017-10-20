import numpy as np
import pandas as pd

import warnings

import pytest

from catwalk.estimators.transformers import CutOff, \
    SubsetWithCategoricals, flatten_list
from catwalk.estimators.classifiers import ScaledLogisticRegression, \
    CatInATreeClassifier, CatInAForestClassifier

from sklearn import linear_model

from sklearn import datasets
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

@pytest.fixture
def data():
    dataset = datasets.load_breast_cancer()
    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=12345)

    return {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}

def test_cutoff_warning():
    X_data = [1, 2, 0.5, 0.7, 100, -1, -23, 0]

    cutoff = CutOff()

    with pytest.warns(UserWarning):
       cutoff.fit_transform(X_data)


def test_cutoff_transformer():
    cutoff = CutOff()

    X_data = [1, 2, 0.5, 0.7, 100, -1, -23, 0]

    assert np.all(cutoff.fit_transform(X_data) == [1, 1, 0.5, 0.7, 1, 0, 0, 0])

def test_cutoff_inside_a_pipeline(data):
    minmax_scaler = preprocessing.MinMaxScaler()
    dsapp_cutoff = CutOff()

    pipeline =Pipeline([
        ('minmax_scaler',minmax_scaler),
        ('dsapp_cutoff', dsapp_cutoff)
    ])

    pipeline.fit(data['X_train'], data['y_train'])

    X_fake_new_data = data['X_test'][-1,:] + 0.5

    mms = preprocessing.MinMaxScaler().fit(data['X_train'])

    assert np.all(( mms.transform(X_fake_new_data) > 1  ) == (pipeline.transform(X_fake_new_data) == 1))

def test_dsapp_lr(data):
    dsapp_lr = ScaledLogisticRegression()
    dsapp_lr.fit(data['X_train'], data['y_train'])

    minmax_scaler = preprocessing.MinMaxScaler()
    dsapp_cutoff = CutOff()
    lr = linear_model.LogisticRegression()

    pipeline =Pipeline([
        ('minmax_scaler',minmax_scaler),
        ('dsapp_cutoff', dsapp_cutoff),
        ('lr', lr)
    ])

    pipeline.fit(data['X_train'], data['y_train'])

    assert np.all(dsapp_lr.predict(data['X_test']) == pipeline.predict(data['X_test']))

def test_flatten_list():
    assert flatten_list([1, [2,3], [4, [5]], [], 6]) == [1,2,3,4,5,6]
    assert flatten_list([]) == []
    assert flatten_list([1,2,3]) == [1,2,3]
    assert flatten_list([[1,2]]) == [1,2]

def test_subset_with_categoricals():
    df = pd.DataFrame({
        'entity_id': [1,2,3,4],
        'as_of_date': ['2012-01-01','2012-01-01','2012-01-01','2012-01-01'],
        'first_entity_id_1y_c1_top_min': [0,1,0,0],
        'first_entity_id_1y_c1_bottom_min': [1,0,0,0],
        'first_entity_id_1y_c1__NULL_min': [0,0,1,0],
        'first_entity_id_1y_a1_sum': [12,7,0,2],
        'first_entity_id_1y_a2_max': [3,1,4,1],
        'second_entity_id_10y_a3_sum': [5,9,2,6],
        'second_entity_id_10y_c3_one_sum': [1,1,0,1],
        'second_entity_id_10y_c3_two_sum': [0,0,1,0],
        'outcome': [0,1,0,0]
        })
    # ensure column order
    df = df[['entity_id', 'as_of_date', 'first_entity_id_1y_c1_top_min', 
             'first_entity_id_1y_c1_bottom_min', 'first_entity_id_1y_c1__NULL_min',
             'first_entity_id_1y_a1_sum', 'first_entity_id_1y_a2_max',
             'second_entity_id_10y_a3_sum', 'second_entity_id_10y_c3_one_sum',
             'second_entity_id_10y_c3_two_sum', 'outcome'
    ]]

    # random seed 0
    sc = SubsetWithCategoricals(
            categoricals=[[0, 1, 2], [6, 7]],
            random_state=0
        )

    samp = sc.fit_transform(df.drop(['entity_id', 'as_of_date', 'outcome'], axis=1).values)

    assert np.all(samp == np.array([  
            [ 0.,  1.,  0.,  1.,  0.],
            [ 1.,  0.,  0.,  1.,  0.],
            [ 0.,  0.,  1.,  0.,  1.],
            [ 0.,  0.,  0.,  1.,  0.]
    ]))
    assert sc.max_features_ == 2
    assert sc.subset_indices == [0, 1, 2, 6, 7]

    # random seed 1
    sc = SubsetWithCategoricals(
            categoricals=[[0, 1, 2], [6, 7]],
            random_state=1
        )

    samp = sc.fit_transform(df.drop(['entity_id', 'as_of_date', 'outcome'], axis=1).values)

    assert np.all(samp == np.array([
            [ 12.,   3.],
            [  7.,   1.],
            [  0.,   4.],
            [  2.,   1.]
    ]))
    assert sc.max_features_ == 2
    assert sc.subset_indices == [3,4]

def test_cat_in_a_tree(data):
    # just for the purposes of testing, assuming several of the columns are categoricals
    categoricals = [[2,3,4], [7,8,9,10,11], [13,14], [22,23,24,25]]

    clf = CatInATreeClassifier(categoricals=categoricals, max_features=7, random_state=12345)
    clf.fit(data['X_train'], data['y_train'])

    assert clf.max_features_ == 7
    assert clf.subset_indices == [0, 7, 8, 9, 10, 11, 12, 16, 19, 21, 27]
    
    pred = clf.predict_proba(data['X_test'])
    assert len(pred) == len(data['y_test'])
    # specific to the breast cancer data...
    assert round(sum([p[1] for p in pred])) == 102


def test_cat_in_a_forest(data):
    # just for the purposes of testing, assuming several of the columns are categoricals
    categoricals = [[2,3,4], [7,8,9,10,11], [13,14], [22,23,24,25]]

    clf = CatInAForestClassifier(categoricals=categoricals, max_features_tree=7, n_estimators=3, random_state=12345)
    clf.fit(data['X_train'], data['y_train'])

    assert clf.estimators_[0].max_features_ == 7
    assert clf.estimators_[0].subset_indices == [0, 1, 6, 12, 13, 14, 18, 22, 23, 24, 25]
    assert clf.estimators_[1].subset_indices == [0, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 21]
    assert clf.estimators_[2].subset_indices == [0, 2, 3, 4, 12, 13, 14, 15, 27, 28]
    
    pred = clf.predict_proba(data['X_test'])
    assert len(pred) == len(data['y_test'])
    # specific to the breast cancer data...
    # even with 
    assert round(sum([p[1] for p in pred])) == 108
