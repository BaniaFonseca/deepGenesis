import pathlib
import tensorflow as tf
DATA_ROOT = pathlib.Path('data_root')
ALL_DATA = pathlib.Path('data_root/all_data')
TEST_DATA = pathlib.Path('data_root/test_data')
TRAIN_DATA = pathlib.Path('data_root/train_data')

if not DATA_ROOT.exists():
    DATA_ROOT.mkdir()

if not ALL_DATA.exists():
    ALL_DATA.mkdir()

if not TEST_DATA.exists():
    TEST_DATA.mkdir()

if not TRAIN_DATA.exists():
    TRAIN_DATA.mkdir()

DATASET_SIZE = len(list(ALL_DATA.glob('*/*')))
TRAIN_SIZE = int(0.7 * DATASET_SIZE)
TEST_SIZE = int(0.3 * DATASET_SIZE)
HEIGHT = 256
WIDTH = 256
CLASS_NUMBER = len(list(item.name for item in ALL_DATA.glob('*/') if item.is_dir()))
