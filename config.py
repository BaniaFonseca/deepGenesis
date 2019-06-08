import pathlib

DATA_ROOT = pathlib.Path('data_root')
ALL_DATA = pathlib.Path('data_root/all_data')
TEST_DATA = pathlib.Path('data_root/test_data')
TRAIN_DATA = pathlib.Path('data_root/train_data')
DARKNET_DIR = pathlib.Path('model/darknet')
DARKNET_DIR_RES = pathlib.Path('model/darknet/resource')
RESNET34_DIR = pathlib.Path('model/resnet34')
RESNET34_DIR_RES = pathlib.Path('model/resnet34/resource')
RESNET50_DIR = pathlib.Path('model/resnet50')
RESNET50_DIR_RES = pathlib.Path('model/resnet50/resource')

if not DATA_ROOT.exists():
    DATA_ROOT.mkdir()

if not ALL_DATA.exists():
    ALL_DATA.mkdir()

if not TEST_DATA.exists():
    TEST_DATA.mkdir()

if not TRAIN_DATA.exists():
    TRAIN_DATA.mkdir()

DATASET_SIZE = len(list(ALL_DATA.glob('empty/*')))
TRAIN_SIZE = int(0.7 * DATASET_SIZE)
TEST_SIZE = int(0.3 * DATASET_SIZE)
HEIGHT = 256
WIDTH = 256
CHANELS = 3
CLASS_NUMBER = len(list(item.name for item in ALL_DATA.glob('*/') if item.is_dir()))
LABEL_NAMES = [item.name for item in ALL_DATA.glob('*/') if item.is_dir()]
LABEL_NAMES.sort()
