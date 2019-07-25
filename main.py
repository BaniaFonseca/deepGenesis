from config import *
from train.train import Train
from dataset.data_processing import  DataProcessing
from dataset.dataset import Dataset
from util.visualize_dataset import VisualizeDataset
import tensorflow as tf
from train.darknet.darknet import TDarknet
from train.resnet34.resnet34 import TResNet34
from train.resnet50.resnet50 import TResNet50
from train.inception_v4.inception_v4 import TInception_v4
from test.test_model import TestModel
import numpy as np

print(tf.__version__)

vs = VisualizeDataset()

train = Train()

td = TDarknet()
tr34 = TResNet34()
tr50 = TResNet50()
ti = TInception_v4()

train.start_training(tr34, retrain=False, model_dir=RESNET34_DIR_RES)
# test_model.test(RESNET34_DIR_RES)

# all = \
#      [str(path) for path in list(ALL_DATA.glob('*/*'))]



dp = DataProcessing()
# dp.process_and_save_data()

ds = Dataset()
# ds.save_datasets_as_npy()


# img1 = dp.read_img_and_clear_noise('data_root/test_data/empty/6.jpeg')
# img2 = dp.read_img_and_clear_noise('data_root/hipotesis/full_2.jpg')
#

# transformatios = [dp.flip, dp.color_inversion,
#                    dp.blur, dp.random_noise, dp.rotate]
# dp.augment(img, all[8])
# ci = list([dp.color_inversion])
# img = ci[0](img)
# img1 = dp. color_inversion(img1)
# img = dp.blur(img)
# dp.save_img('data_root/hipotesis/full_1.jpg', img1)
# dp.save_img('data_root/hipotesis/full_2.jpg', img2)

# print(img.shape)
# vs.show_img(img1)
#vs.show_img(img2)



# ti.train()
#ti.test()

# tr34.test()
#td.train()
#td.test()

#tr50.train()
#tr50.test()

#validationtest = ds.load_validationtest()
#testset = ds.load_testset()



# images, labels = ds.load_trainset()
# vs.show_images(images=images, labels=labels, cols=6, rows=3)

#testset = ds.get_testset()
#vs.show_images(dataset=testset, cols=1, rows=1)

#trainset = ds.generate_dataset('processed_data/train')
#print(trainset)
#train = Train(data_root='data')

#vs.show_mean_width_heigth(path='processed_data/train')
#vs.show_mean_aspect_ratio(path='preprocessed_data')


#train.start_training()
