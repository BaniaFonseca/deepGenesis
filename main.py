from config import *
from train.train import Train
from dataset.data_processing import  DataProcessing
from dataset.dataset import Dataset
from util.visualize_dataset import VisualizeDataset
import tensorflow as tf
from train.darknet.darknet import TDarknet
from train.resnet34.resnet34 import TResNet34
from train.resnet50.resnet50 import TResNet50

print(tf.__version__)

td = TDarknet()
tr34 = TResNet34()
tr50 = TResNet50()

#tr34.train()
#tr34.test()
#td.train()
#td.test()

#tr50.train()
#tr50.test()

dp = DataProcessing()
#dp.process_data()

vs = VisualizeDataset()

ds = Dataset()

#images, labels = ds.get_trainset(name='ori*')
#vs.show_images(images=images, labels=labels, cols=6, rows=3)

#testset = ds.get_testset()
#vs.show_images(dataset=testset, cols=1, rows=1)

#trainset = ds.generate_dataset('processed_data/train')
#print(trainset)
#train = Train(data_root='data')

#vs.show_mean_width_heigth(path='processed_data/train')
#vs.show_mean_aspect_ratio(path='preprocessed_data')


#train.start_training()
