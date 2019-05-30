from train.train import Train
from dataset.data_processing import  DataProcessing
from dataset.dataset import Dataset
from util.visualize_dataset import VisualizeDataset
import tensorflow as tf

#tf.compat.v1.enable_eager_execution()

ds = Dataset()

trainset = ds.generate_dataset('processed_data/train')
#print(trainset)
#train = Train(data_root='data')

vs = VisualizeDataset()
#vs.show_mean_width_heigth(path='processed_data/train')
#vs.show_mean_aspect_ratio(path='preprocessed_data')

vs.show_images(dataset=trainset)
#dp = DataProcessing(preprocessed_data_root='preprocessed_data', processed_data_root='processed_data')

#train.start_training()
