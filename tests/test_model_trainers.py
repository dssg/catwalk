import boto3
import pandas
import pickle
import testing.postgresql
import datetime
import sqlalchemy
import unittest
from unittest.mock import patch

from moto import mock_s3
from sqlalchemy import create_engine
from catwalk.db import ensure_db
from catwalk.utils import model_cache_key

from catwalk.model_trainers import ModelTrainer
from catwalk.storage import InMemoryModelStorageEngine,\
    S3ModelStorageEngine, InMemoryMatrixStore


def test_model_trainer():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)

        grid_config = {
            'sklearn.linear_model.LogisticRegression': {
                'C': [0.00001, 0.0001],
                'penalty': ['l1', 'l2'],
                'random_state': [2193]
            }
        }

        with mock_s3():
            s3_conn = boto3.resource('s3')
            s3_conn.create_bucket(Bucket='econ-dev')

            # create training set
            matrix = pandas.DataFrame.from_dict({
                'entity_id': [1, 2],
                'feature_one': [3, 4],
                'feature_two': [5, 6],
                'label': ['good', 'bad']
            })
            metadata = {
                'beginning_of_time': datetime.date(2012, 12, 20),
                'end_time': datetime.date(2016, 12, 20),
                'label_name': 'label',
                'label_window': '1y',
                'metta-uuid': '1234',
                'feature_names': ['ft1', 'ft2'],
                'indices': ['entity_id'],
            }
            project_path = 'econ-dev/inspections'
            model_storage_engine = S3ModelStorageEngine(s3_conn, project_path)
            trainer = ModelTrainer(
                project_path=project_path,
                experiment_hash=None,
                model_storage_engine=model_storage_engine,
                db_engine=engine,
                model_group_keys=['label_name', 'label_window'],
                feature_config=[]
            )
            matrix_store = InMemoryMatrixStore(matrix, metadata)
            model_ids = trainer.train_models(
                grid_config=grid_config,
                misc_db_parameters=dict(),
                matrix_store=matrix_store
            )

            # assert
            # 1. that the models and feature importances table entries are present
            records = [
                row for row in
                engine.execute('select * from results.feature_importances')
            ]
            assert len(records) == 4 * 2  # maybe exclude entity_id? yes

            records = [
                row for row in
                engine.execute('select model_hash from results.models')
            ]
            assert len(records) == 4

            cache_keys = [
                model_cache_key(project_path, model_row[0], s3_conn)
                for model_row in records
            ]

            # 2. that the model groups are distinct
            records = [
                row for row in
                engine.execute('select distinct model_group_id from results.models')
            ]
            assert len(records) == 4

            # 3. that all four models are cached
            model_pickles = [
                pickle.loads(cache_key.get()['Body'].read())
                for cache_key in cache_keys
            ]
            assert len(model_pickles) == 4
            assert len([x for x in model_pickles if x is not None]) == 4

            # 4. that their results can have predictions made on it
            test_matrix = pandas.DataFrame.from_dict({
                'entity_id': [3, 4],
                'feature_one': [4, 4],
                'feature_two': [6, 5],
            })

            test_matrix = InMemoryMatrixStore(matrix=test_matrix, metadata=metadata).matrix

            for model_pickle in model_pickles:
                predictions = model_pickle.predict(test_matrix)
                assert len(predictions) == 2

            # 5. when run again, same models are returned
            new_model_ids = trainer.train_models(
                grid_config=grid_config,
                misc_db_parameters=dict(),
                matrix_store=matrix_store
            )
            assert len([
                row for row in
                engine.execute('select model_hash from results.models')
            ]) == 4
            assert model_ids == new_model_ids

            # 6. if replace is set, update non-unique attributes and feature importances
            max_batch_run_time = [
                row[0] for row in
                engine.execute('select max(batch_run_time) from results.models')
            ][0]
            trainer = ModelTrainer(
                project_path=project_path,
                experiment_hash=None,
                model_storage_engine=model_storage_engine,
                db_engine=engine,
                model_group_keys=['label_name', 'label_window'],
                feature_config=[],
                replace=True
            )
            new_model_ids = trainer.train_models(
                grid_config=grid_config,
                misc_db_parameters=dict(),
                matrix_store=matrix_store,
            )
            assert model_ids == new_model_ids
            assert [
                row['model_id'] for row in
                engine.execute('select model_id from results.models order by 1 asc')
            ] == model_ids
            new_max_batch_run_time = [
                row[0] for row in
                engine.execute('select max(batch_run_time) from results.models')
            ][0]
            assert new_max_batch_run_time > max_batch_run_time

            records = [
                row for row in
                engine.execute('select * from results.feature_importances')
            ]
            assert len(records) == 4 * 2  # maybe exclude entity_id? yes

            # 7. if the cache is missing but the metadata is still there, reuse the metadata
            for row in engine.execute('select model_hash from results.models'):
                model_storage_engine.get_store(row[0]).delete()
            new_model_ids = trainer.train_models(
                grid_config=grid_config,
                misc_db_parameters=dict(),
                matrix_store=matrix_store
            )
            assert model_ids == sorted(new_model_ids)

            # 8. that the generator interface works the same way
            new_model_ids = trainer.generate_trained_models(
                grid_config=grid_config,
                misc_db_parameters=dict(),
                matrix_store=matrix_store
            )
            assert model_ids == \
                sorted([model_id for model_id in new_model_ids])


def test_model_trainer_categoricals():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)

        grid_config = {
            'catwalk.estimators.classifiers.CatInAForestClassifier': {
                'max_features_tree': [3],
                'n_estimators': [3],
                'random_state': [2193]
            }
        }

        with mock_s3():
            s3_conn = boto3.resource('s3')
            s3_conn.create_bucket(Bucket='econ-dev')

            feature_config = [
                {
                    'prefix': 'first',
                    'aggregates': [
                        {'quantity': 'a1', 'metrics': ['sum']},
                        {'quantity': 'a2', 'metrics': ['max']}
                    ],
                    'categoricals': [
                        {'column': 'c1', 'choices': ['top', 'bottom', 'charm', 'strange'], 'metrics': ['min']}
                    ],
                    'intervals': ['1y'],
                    'groups': ['entity_id']
                },
                {
                    'prefix': 'second',
                    'aggregates': [
                        {'quantity': 'a3', 'metrics': ['sum']}
                    ],
                    'categoricals': [
                        {'column': 'c3', 'choices': ['one', 'two'], 'metrics': ['sum']}
                    ],
                    'intervals': ['10y'],
                    'groups': ['entity_id']
                }
            ]

            # create training set
            matrix = pandas.DataFrame.from_dict({
                'entity_id': [1,2,3,4],
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
            matrix = matrix[['entity_id', 'first_entity_id_1y_c1_top_min', 
                     'first_entity_id_1y_c1_bottom_min', 'first_entity_id_1y_c1__NULL_min',
                     'first_entity_id_1y_a1_sum', 'first_entity_id_1y_a2_max',
                     'second_entity_id_10y_a3_sum', 'second_entity_id_10y_c3_one_sum',
                     'second_entity_id_10y_c3_two_sum', 'outcome'
            ]]
            metadata = {
                'beginning_of_time': datetime.date(2012, 12, 20),
                'end_time': datetime.date(2016, 12, 20),
                'label_name': 'outcome',
                'label_window': '1y',
                'metta-uuid': '1234',
                'feature_names': ['first_entity_id_1y_c1_top_min', 
                     'first_entity_id_1y_c1_bottom_min', 'first_entity_id_1y_c1__NULL_min',
                     'first_entity_id_1y_a1_sum', 'first_entity_id_1y_a2_max',
                     'second_entity_id_10y_a3_sum', 'second_entity_id_10y_c3_one_sum',
                     'second_entity_id_10y_c3_two_sum'
                ],
                'indices': ['entity_id'],
            }
            project_path = 'econ-dev/inspections'
            model_storage_engine = S3ModelStorageEngine(s3_conn, project_path)
            trainer = ModelTrainer(
                project_path=project_path,
                experiment_hash=None,
                model_storage_engine=model_storage_engine,
                db_engine=engine,
                model_group_keys=['label_name', 'label_window'],
                feature_config=feature_config
            )
            matrix_store = InMemoryMatrixStore(matrix, metadata)
            model_ids = trainer.train_models(
                grid_config=grid_config,
                misc_db_parameters=dict(),
                matrix_store=matrix_store
            )

            # assert categoricals were properly detected and passed to model
            records = [
                row for row in
                engine.execute('select model_hash from results.models')
            ]

            cache_keys = [
                model_cache_key(project_path, model_row[0], s3_conn)
                for model_row in records
            ]

            model_pickles = [
                pickle.loads(cache_key.get()['Body'].read())
                for cache_key in cache_keys
            ]

            assert sorted([sorted(c) for c in model_pickles[0].categoricals]) == [[0, 1, 2], [6, 7]]


def test_n_jobs_not_new_model():
    grid_config = {
        'sklearn.ensemble.AdaBoostClassifier': {
            'n_estimators': [10, 100, 1000]
        },
        'sklearn.ensemble.RandomForestClassifier': {
            'n_estimators': [10, 100],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [5, 10, 15, 20],
            'criterion': ['gini', 'entropy'],
            'n_jobs': [12, 24],
        }
    }

    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        ensure_db(engine)
        with mock_s3():
            s3_conn = boto3.resource('s3')
            s3_conn.create_bucket(Bucket='econ-dev')
            trainer = ModelTrainer(
                project_path='econ-dev/inspections',
                experiment_hash=None,
                model_storage_engine=S3ModelStorageEngine(s3_conn, 'econ-dev/inspections'),
                db_engine=engine,
                model_group_keys=['label_name', 'label_window'],
                feature_config=[]
            )

            matrix = pandas.DataFrame.from_dict({
                'entity_id': [1, 2],
                'feature_one': [3, 4],
                'feature_two': [5, 6],
                'label': ['good', 'bad']
            })
            train_tasks = trainer.generate_train_tasks(
                grid_config,
                dict(),
                InMemoryMatrixStore(matrix, {
                    'label_window': '1d',
                    'end_time': datetime.datetime.now(),
                    'beginning_of_time': datetime.date(2012, 12, 20),
                    'label_name': 'label',
                    'metta-uuid': '1234',
                    'feature_names': ['ft1', 'ft2'],
                    'indices': ['entity_id'],
                })
            )
            assert len(train_tasks) == 35 # 32+3, would be (32*2)+3 if we didn't remove
            assert len([
                task for task in train_tasks
                if 'n_jobs' in task['parameters']
            ]) == 32

            for train_task in train_tasks:
                trainer.process_train_task(**train_task)

            for row in engine.execute(
                'select model_parameters from results.model_groups'
            ):
                assert 'n_jobs' not in row[0]


class RetryTest(unittest.TestCase):
    def test_retry_max(self):
        grid_config = {
            'sklearn.ensemble.AdaBoostClassifier': {
                'n_estimators': [10]
            },
        }

        engine = None
        trainer = None
        # set up a basic model training run
        # TODO abstract the setup of a basic model training run where
        # we don't worry about the specific values used? it would make
        # tests like this require a bit less noise to read past
        with testing.postgresql.Postgresql() as postgresql:
            engine = create_engine(postgresql.url())
            ensure_db(engine)
            trainer = ModelTrainer(
                project_path='econ-dev/inspections',
                experiment_hash=None,
                model_storage_engine=InMemoryModelStorageEngine(project_path=''),
                db_engine=engine,
                model_group_keys=['label_name', 'label_window'],
                feature_config=[]
            )

            matrix = pandas.DataFrame.from_dict({
                'entity_id': [1, 2],
                'feature_one': [3, 4],
                'feature_two': [5, 6],
                'label': ['good', 'bad']
            })
            matrix_store = InMemoryMatrixStore(matrix, {
                'label_window': '1d',
                'end_time': datetime.datetime.now(),
                'beginning_of_time': datetime.date(2012, 12, 20),
                'label_name': 'label',
                'metta-uuid': '1234',
                'feature_names': ['ft1', 'ft2'],
                'indices': ['entity_id'],
            })
        # the postgres server goes out of scope here and thus no longer exists
        with patch('time.sleep') as time_mock:
            with self.assertRaises(sqlalchemy.exc.OperationalError):
                trainer.train_models(grid_config, dict(), matrix_store)
            # we want to make sure that we are using the retrying module sanely
            # as opposed to matching the exact # of calls specified by the code
            assert len(time_mock.mock_calls) > 5

    def test_retry_recovery(self):
        grid_config = {
            'sklearn.ensemble.AdaBoostClassifier': {
                'n_estimators': [10]
            },
        }

        engine = None
        trainer = None
        port = None
        with testing.postgresql.Postgresql() as postgresql:
            port = postgresql.settings['port']
            engine = create_engine(postgresql.url())
            ensure_db(engine)
            trainer = ModelTrainer(
                project_path='econ-dev/inspections',
                experiment_hash=None,
                model_storage_engine=InMemoryModelStorageEngine(project_path=''),
                db_engine=engine,
                model_group_keys=['label_name', 'label_window'],
                feature_config=[]
            )

            matrix = pandas.DataFrame.from_dict({
                'entity_id': [1, 2],
                'feature_one': [3, 4],
                'feature_two': [5, 6],
                'label': ['good', 'bad']
            })
            matrix_store = InMemoryMatrixStore(matrix, {
                'label_window': '1d',
                'end_time': datetime.datetime.now(),
                'beginning_of_time': datetime.date(2012, 12, 20),
                'label_name': 'label',
                'metta-uuid': '1234',
                'feature_names': ['ft1', 'ft2'],
                'indices': ['entity_id'],
            })

        # start without a database server
        # then bring it back up after the first sleep
        # use self so it doesn't go out of scope too early and shut down
        self.new_server = None
        def replace_db(arg):
            self.new_server = testing.postgresql.Postgresql(port=port)
            engine = create_engine(self.new_server.url())
            ensure_db(engine)
        with patch('time.sleep') as time_mock:
            time_mock.side_effect = replace_db
            try:
                trainer.train_models(grid_config, dict(), matrix_store)
            finally:
                if self.new_server is not None:
                    self.new_server.stop()
            assert len(time_mock.mock_calls) == 1
