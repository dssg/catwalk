from catwalk.utils import filename_friendly_hash, \
    save_experiment_and_get_hash, \
    sort_predictions_and_labels, \
    bag_of_cats, find_cats
from catwalk.db import ensure_db
from sqlalchemy import create_engine
import testing.postgresql
import datetime
import logging
import re
import pandas as pd


def test_filename_friendly_hash():
    data = {
        'stuff': 'stuff',
        'other_stuff': 'more_stuff',
        'a_datetime': datetime.datetime(2015, 1, 1),
        'a_date': datetime.date(2016, 1, 1),
        'a_number': 5.0
    }
    output = filename_friendly_hash(data)
    assert isinstance(output, str)
    assert re.match('^[\w]+$', output) is not None

    # make sure ordering keys differently doesn't change the hash
    new_output = filename_friendly_hash({
        'other_stuff': 'more_stuff',
        'stuff': 'stuff',
        'a_datetime': datetime.datetime(2015, 1, 1),
        'a_date': datetime.date(2016, 1, 1),
        'a_number': 5.0
    })
    assert new_output == output

    # make sure new data hashes to something different
    new_output = filename_friendly_hash({
        'stuff': 'stuff',
        'a_number': 5.0
    })
    assert new_output != output


def test_filename_friendly_hash_stability():
    nested_data = {
        'one': 'two',
        'three': {
            'four': 'five',
            'six': 'seven'
        }
    }
    output = filename_friendly_hash(nested_data)
    # 1. we want to make sure this is stable across different runs
    # so hardcode an expected value
    assert output == '9a844a7ebbfd821010b1c2c13f7391e6'
    other_nested_data = {
        'one': 'two',
        'three': {
            'six': 'seven',
            'four': 'five'
        }
    }
    new_output = filename_friendly_hash(other_nested_data)
    assert output == new_output


def test_save_experiment_and_get_hash():
    # no reason to make assertions on the config itself, use a basic dict
    experiment_config = {'one': 'two'}
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)
        exp_hash = save_experiment_and_get_hash(experiment_config, engine)
        assert isinstance(exp_hash, str)
        new_hash = save_experiment_and_get_hash(experiment_config, engine)
        assert new_hash == exp_hash

def test_sort_predictions_and_labels():
    predictions = [
        0.5,
        0.4,
        0.6,
        0.5,
    ]

    labels = [
        False,
        False,
        True,
        True
    ]

    sorted_predictions, sorted_labels = sort_predictions_and_labels(
        predictions,
        labels,
        8
    )
    assert sorted_predictions == (0.6, 0.5, 0.5, 0.4)
    assert sorted_labels == (True, True, False, False)


    sorted_predictions, sorted_labels = sort_predictions_and_labels(
        predictions,
        labels,
        12345
    )
    assert sorted_predictions == (0.6, 0.5, 0.5, 0.4)
    assert sorted_labels == (True, False, True, False)

def test_bag_of_cats():
    feature_config = [
        {
            'prefix': 'first',
            'aggregates': [
                {'quantity': 'a1', 'metrics': ['min', 'max']}
            ],
            'categoricals': [
                {'column': 'c1', 'choices': ['top', 'bottom', 'charm', 'strange'], 'metrics': ['min']},
                {'column': 'c2', 'choices': ['up', 'down'], 'metrics': ['sum', 'max']}
            ],
            'intervals': ['1y', '5y'],
            'groups': ['entity_id']
        },
        {
            'prefix': 'second',
            'categoricals': [
                {'column': 'c3', 'choices': ['one', 'two'], 'metrics': ['sum']},
                {'column': 'c4', 'choices': ['three', 'four'], 'metrics': ['max']}
            ],
            'intervals': ['1y', '10y'],
            'groups': ['entity_id']
        },
        {
            'prefix': 'third',
            'aggregates': [
                {'quantity': 'a2', 'metrics': ['min', 'max']}
            ],
            'intervals': ['6month'],
            'groups': ['entity_id']
        }
    ]

    cat_regex = set(bag_of_cats(feature_config))

    assert cat_regex == set([
        r'^first_entity_id_1y_c1_(.*)_min$', r'^first_entity_id_5y_c1_(.*)_min$', 
        r'^first_entity_id_1y_c2_(.*)_sum$', r'^first_entity_id_1y_c2_(.*)_max$', 
        r'^first_entity_id_5y_c2_(.*)_sum$', r'^first_entity_id_5y_c2_(.*)_max$', 
        r'^second_entity_id_1y_c3_(.*)_sum$', r'^second_entity_id_10y_c3_(.*)_sum$', 
        r'^second_entity_id_1y_c4_(.*)_max$', r'^second_entity_id_10y_c4_(.*)_max$'
    ])

def test_find_cats():
    cat_regex = [r'^first_entity_id_1y_c1_(.*)_min$', r'^second_entity_id_10y_c3_(.*)_sum$']
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

    cat_cols = find_cats(df.columns.values, cat_regex)

    assert cat_cols == [[0, 1, 2], [6, 7]]
