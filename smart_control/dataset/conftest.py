"""Tests for the Smart Buildings Dataset.

Includes high fidelity tests to download the dataset and verify its structure.

It takes around two minutes to download and unzip the data, so we are skipping
dataset tests by default, to keep the build fast. But you can trigger a download
by setting the `TEST_DATASET_DOWNLOAD` environment variable to 'true'.

Downloaded data will not get cleared after tests run, so we can use it in
subsequent test runs without needing to re-download. This allows developers to
run dataset tests fairly quickly on their local machines. When the dataset
already exists locally, it only takes around five seconds to load.

The dataset tests will be run if the data is being downloaded, or if there is
existing local data.

Downloaded data will not get cleared by default before tests run, but you can
force a clean up and fresh download by setting the `CLEAR_TEST_DATASET_DOWNLOAD`
environment variable to 'true'.
"""

import os
import shutil

from dotenv import load_dotenv
import pytest

from smart_control.dataset.dataset import BuildingDataset
from smart_control.dataset.dataset import DATA_DIR
from smart_control.dataset.partition import BuildingDatasetPartition

load_dotenv()

# whether or not to download the dataset:
TEST_DATASET_DOWNLOAD = bool(
    os.getenv('TEST_DATASET_DOWNLOAD', default='false').lower() == 'true'
)
# whether or not to delete existing local data before downloading:
CLEAR_TEST_DATASET_DOWNLOAD = bool(
    os.getenv('CLEAR_TEST_DATASET_DOWNLOAD', default='false').lower() == 'true'
)

DATASET_DIRPATH = os.path.join(DATA_DIR, 'sb1')
ZIP_FILEPATH = os.path.join(DATA_DIR, 'sb1.zip')

# whether or not to run dataset tests:
TEST_DATASET = bool(TEST_DATASET_DOWNLOAD or os.path.isdir(DATASET_DIRPATH))
SKIP_REASON = 'Skip large download by default.'


def cleanup_files():
  print('Deleting dataset files...')

  if os.path.isfile(ZIP_FILEPATH):
    os.remove(ZIP_FILEPATH)

  if os.path.isdir(DATASET_DIRPATH):
    shutil.rmtree(DATASET_DIRPATH)


# PYTEST FIXTURES


@pytest.fixture(scope='session')
def dataset():
  """Session-scoped pytest fixture for an example dataset.
  Will be executed only once, and can be shared across multiple test files.
  """
  if TEST_DATASET_DOWNLOAD and CLEAR_TEST_DATASET_DOWNLOAD:
    cleanup_files()

  print('Initializing the dataset fixture...')
  return BuildingDataset(dataset_id='sb1', download=TEST_DATASET_DOWNLOAD)


@pytest.fixture(scope='session')
def partition(dataset):  # pylint: disable=redefined-outer-name
  """Session-scoped pytest fixture for an example dataset partition.
  Will be executed only once, and can be shared across multiple test files.
  """
  return BuildingDatasetPartition(dataset=dataset, partition_id='2022_a')


# SHIMS TO GET PYTEST FIXTURES TO WORK WITH UNITTEST CLASSES :-)


@pytest.fixture(scope='class')
def set_dataset(request, dataset):  # pylint: disable=redefined-outer-name
  """
  A class-scoped fixture that takes the result of the 'dataset' fixture and
  injects it into the test class as `cls.ds`.

  Use by decorating your test class with:

    @pytest.mark.usefixtures('set_dataset')

  NOTE: the injection happens AFTER the setUp methods run in the test class.
  """
  if request.cls:
    request.cls.ds = dataset


@pytest.fixture(scope='class')
def set_partition(request, partition):  # pylint: disable=redefined-outer-name, line-too-long
  """
  A class-scoped fixture that takes the result of the 'partition' fixture and
  injects it into the test class as `cls.partition`.

  Use by decorating your test class with:

    @pytest.mark.usefixtures('set_partition')

  NOTE: the injection happens AFTER the setUp methods run in the test class.
  """
  if request.cls:
    request.cls.partition = partition
