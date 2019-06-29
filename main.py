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

print(tf.__version__)

td = TDarknet()
tr34 = TResNet34()
tr50 = TResNet50()
ti = TInception_v4()

dp = DataProcessing()
# dp.process_data()

# ti.train()
#ti.test()
tr34.train(retrain=False, test=True)
# tr34.test()
#td.train()
#td.test()

#tr50.train()
#tr50.test()

ds = Dataset()
#ds.save_datasets_as_npy()

vs = VisualizeDataset()

#validationtest = ds.load_validationtest()
#testset = ds.load_testset()



#images, labels = ds.load_trainset()
#vs.show_images(images=images, labels=labels, cols=6, rows=3)

#testset = ds.get_testset()
#vs.show_images(dataset=testset, cols=1, rows=1)

#trainset = ds.generate_dataset('processed_data/train')
#print(trainset)
#train = Train(data_root='data')

#vs.show_mean_width_heigth(path='processed_data/train')
#vs.show_mean_aspect_ratio(path='preprocessed_data')


#train.start_training()
