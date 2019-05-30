from train.train import Train
from dataset.data_processing import  DataProcessing
from util.visualize_dataset import VisualizeDataset

#train = Train(data_root='data')

#vs = VisualizeDataset(data_root='data')
#vs.show_mean_width_heigth(preprocess=False)
#vs.show_images(augmentend_data_root='augmented_data')
aap = DataProcessing(preprocessed_data_root='preprocessed_data', processed_data_root='processed_data')

#train.start_training()
