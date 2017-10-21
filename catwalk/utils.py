import datetime
import pickle
import tempfile
import hashlib
import botocore
import pandas
import random
import yaml
import json
from results_schema import Experiment, Model
from retrying import retry
from sqlalchemy.orm import sessionmaker
import sqlalchemy
import csv
import postgres_copy
from itertools import product
import re


def split_s3_path(path):
    """
    Args:
        path: a string representing an s3 path including a bucket
            (bucket_name/prefix/prefix2)
    Returns:
        A tuple containing the bucket name and full prefix)
    """
    return path.split('/', 1)


def upload_object_to_key(obj, cache_key):
    """Pickles object and uploads it to the given s3 key

    Args:
        obj (object) any picklable Python object
        cache_key (boto3.s3.Object) an s3 key
    """
    with tempfile.NamedTemporaryFile('w+b') as f:
        pickle.dump(obj, f)
        f.seek(0)
        cache_key.upload_file(f.name)


def download_object(cache_key):
    with tempfile.NamedTemporaryFile() as f:
        cache_key.download_fileobj(f)
        f.seek(0)
        return pickle.load(f)


def model_cache_key(project_path, model_id, s3_conn):
    """Generates an s3 key for a given model_id

    Args:
        model_id (string) a unique model id

    Returns:
        (boto3.s3.Object) an s3 key, which may or may not have contents
    """
    bucket_name, prefix = split_s3_path(project_path)
    path = '/'.join([prefix, 'trained_models', model_id])
    return s3_conn.Object(bucket_name, path)


def key_exists(key):
    try:
        key.load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise
    else:
        return True


def get_matrix_and_metadata(matrix_path, metadata_path):
    """Retrieve a matrix in hdf format and
    metadata about the matrix in yaml format

    Returns: (tuple) matrix, metadata
    """
    matrix = pandas.read_hdf(matrix_path)
    with open(metadata_path) as f:
        metadata = yaml.load(f)
    return matrix, metadata


def filename_friendly_hash(inputs):
    def dt_handler(x):
        if isinstance(x, datetime.datetime) or isinstance(x, datetime.date):
            return x.isoformat()
        raise TypeError("Unknown type")
    return hashlib.md5(
        json.dumps(inputs, default=dt_handler, sort_keys=True)
            .encode('utf-8')
    ).hexdigest()


def retry_if_db_error(exception):
    return isinstance(exception, sqlalchemy.exc.OperationalError)


DEFAULT_RETRY_KWARGS = {
    'retry_on_exception': retry_if_db_error,
    'wait_exponential_multiplier': 1000, # wait 2^x*1000ms between each retry
    'stop_max_attempt_number': 14,
    # with this configuration, last wait will be ~2 hours
    # for a total of ~4.5 hours waiting
}


db_retry = retry(**DEFAULT_RETRY_KWARGS)


@db_retry
def save_experiment_and_get_hash(config, db_engine):
    experiment_hash = filename_friendly_hash(config)
    session = sessionmaker(bind=db_engine)()
    session.merge(Experiment(
        experiment_hash=experiment_hash,
        config=config
    ))
    session.commit()
    session.close()
    return experiment_hash


class Batch:
    # modified from
    # http://codereview.stackexchange.com/questions/118883/split-up-an-iterable-into-batches
    def __init__(self, iterable, limit=None):
        self.iterator = iter(iterable)
        self.limit = limit
        try:
            self.current = next(self.iterator)
        except StopIteration:
            self.on_going = False
        else:
            self.on_going = True

    def group(self):
        yield self.current
        # start enumerate at 1 because we already yielded the last saved item
        for num, item in enumerate(self.iterator, 1):
            self.current = item
            if num == self.limit:
                break
            yield item
        else:
            self.on_going = False

    def __iter__(self):
        while self.on_going:
            yield self.group()


def sort_predictions_and_labels(predictions_proba, labels, sort_seed):
    random.seed(sort_seed)
    predictions_proba_sorted, labels_sorted = zip(*sorted(
        zip(predictions_proba, labels),
        key=lambda pair: (pair[0], random.random()), reverse=True)
    )
    return predictions_proba_sorted, labels_sorted


@db_retry
def retrieve_model_id_from_hash(db_engine, model_hash):
    """Retrieves a model id from the database that matches the given hash

    Args:
        db_engine (sqlalchemy.engine) A database engine
        model_hash (str) The model hash to lookup

    Returns: (int) The model id (if found in DB), None (if not)
    """
    session = sessionmaker(bind=db_engine)()
    try:
        saved = session.query(Model)\
            .filter_by(model_hash=model_hash)\
            .one_or_none()
        return saved.model_id if saved else None
    finally:
        session.close()


@db_retry
def save_db_objects(db_engine, db_objects):
    """Saves a collection of SQLAlchemy model objects to the database using a COPY command

    Args:
        db_engine (sqlalchemy.engine)
        db_objects (list) SQLAlchemy model objects, corresponding to a valid table
    """
    with tempfile.TemporaryFile(mode='w+') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for db_object in db_objects:
            writer.writerow([
                getattr(db_object, col.name)
                for col in db_object.__table__.columns
            ])
        f.seek(0)
        postgres_copy.copy_from(f, type(db_objects[0]), db_engine, format='csv')


# Two methods for identifying and grouping categorical columns
def bag_of_cats(feature_config):
    """
    Parse a feature config to create regex patterns to match
    categorical columns. Note that this assumes there's no
    column name truncation
    """
    cats_regex = []
    for fg in feature_config:
        prefix = fg['prefix']
        groups = fg['groups']
        intervals = fg['intervals']
        cats = fg.get('categoricals', [])
        for cat in cats:
            col = cat['column']
            metrics = cat['metrics']

            for group, interval, metric in product(
                groups, intervals, metrics
                ):
                cats_regex.append(r'^%s_%s_%s_%s_(.*)_%s$' % (
                    prefix, group, interval, col, metric
                ))

    return cats_regex


# assumes no column name truncation!!
def find_cats(matrix_cols, cats_regex, exclude_cols=None):
    """
    Assign matrix columns (by their numerical indices) to groups
    of categoricals based on matching to a regex pattern

    Note that groupings of imputed columns along with their
    underlying columns will be included in the returned result
    as well.
    """

    # be sure we exclude entity id, date, and label
    if exclude_cols is None:
        exclude_cols = ['entity_id', 'as_of_date', 'outcome']
    feature_cols = [c for c in matrix_cols if c not in exclude_cols]

    # add in regex to make sure imputed flags always come along with
    # their reference columns
    # TODO: maybe return these as a separate list to allow models to
    #       treat them differently than categoricals.
    imp_regex = [
        r'^%s(_imp)?$' % col[:-4] for col in matrix_cols if col[-4:] == '_imp'
    ]
    cats_regex += imp_regex

    # We want the sets of numberical indices of columns that match our
    # categorical patterns, so loop trough the column names then through
    # the patterns, checking each one for a match. Here, `cats_dict`
    # will act as a collector to hold the matches associated with each
    # pattern. Note that if a column matches two patterns, it will get 
    # assigned to the first categorical that matches, though this 
    # shouldn't happen if the regex is matching the full string...
    cats_dict = {r:[] for r in cats_regex}
    for i, fc in enumerate(feature_cols):
        for regex in cats_regex:
            m = re.match(regex, fc)
            if m is not None:
                cats_dict[regex].append(i)
                break

    # collapse the dict into a list of lists to return
    return [v for v in cats_dict.values() if len(v) > 0]
