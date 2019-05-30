from train.train import Train
from dataset.data_processing import  DataProcessing
from util.visualize_dataset import VisualizeDataset

#train = Train(data_root='data')

vs = VisualizeDataset(data_root='processed_data')
#vs.show_mean_width_heigth(preprocess=False)
vs.show_images(processed_data_root='processed_data/train')
#dp = DataProcessing(preprocessed_data_root='preprocessed_data', processed_data_root='processed_data')

#train.start_training()
