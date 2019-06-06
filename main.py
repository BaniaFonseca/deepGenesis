from config import *
from train.train import Train
from dataset.data_processing import  DataProcessing
from dataset.dataset import Dataset
from util.visualize_dataset import VisualizeDataset
import tensorflow as tf
from train.yolo.yolo import TYolo
from train.retinanet.retinanet import TRetinaNet


ty = TYolo()
tr = TRetinaNet()

#tr.train()
#ty.train()
ty.test()

dp = DataProcessing()
#dp.process_data()

vs = VisualizeDataset()

ds = Dataset()

#images, labels = ds.get_trainset(name='zoo*')
#vs.show_images(images=images, labels=labels, cols=6, rows=3)

#testset = ds.get_testset()
#vs.show_images(dataset=testset, cols=1, rows=1)

#trainset = ds.generate_dataset('processed_data/train')
#print(trainset)
#train = Train(data_root='data')

#vs.show_mean_width_heigth(path='processed_data/train')
#vs.show_mean_aspect_ratio(path='preprocessed_data')


#train.start_training()
