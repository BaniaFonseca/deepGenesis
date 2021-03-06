import pathlib

DATA_ROOT = pathlib.Path('data_root')
ALL_DATA = pathlib.Path('data_root/all_data')
VALIDATION_DATA = pathlib.Path('data_root/validation_data')
TEST_DATA = pathlib.Path('data_root/test_data')
TRAIN_DATA = pathlib.Path('data_root/train_data')
DATA_ROOTS = [TRAIN_DATA,VALIDATION_DATA,TEST_DATA]

DATASET_SEED = 67280421310721
K_FOLD_SEED = 2137  # [(1, 7), (6, 1097), (11, 2017), (16, 2027), (21, 2087), (26, 2137)]
TRAIN_PROPORSION = 1.0
TEST_PROPORSION = 0.0
VALIDATION_PROPORSION = 0.0

DARKNET_DIR = pathlib.Path('model/darknet')
DARKNET_DIR_RES = pathlib.Path('model/darknet/resource')

RESNET34_DIR = pathlib.Path('model/resnet34')
RESNET34_DIR_RES = pathlib.Path('model/resnet34/resource')

RESNET50_DIR = pathlib.Path('model/resnet50')
RESNET50_DIR_RES = pathlib.Path('model/resnet50/resource')

INCEPTION_V4_DIR = pathlib.Path('model/inception_v4')
INCEPTION_V4_DIR_RES = pathlib.Path('model/inception_v4/resource')

if not DATA_ROOT.exists():
    DATA_ROOT.mkdir()

if not ALL_DATA.exists():
    ALL_DATA.mkdir()

if not TEST_DATA.exists():
    TEST_DATA.mkdir()

if not VALIDATION_DATA.exists():
    VALIDATION_DATA.mkdir()

if not TRAIN_DATA.exists():
    TRAIN_DATA.mkdir()

HEIGHT = 256
WIDTH = 256
CHANELS = 3

CLASS_NUMBER = len(list(item.name for item in ALL_DATA.glob('*/') if item.is_dir()))
LABEL_NAMES = [item.name for item in ALL_DATA.glob('*/') if item.is_dir()]
LABEL_NAMES.sort()
