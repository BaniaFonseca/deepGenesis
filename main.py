from train.train import Train
from util.visualize_dataset import VisualizeDataset

train = Train(data_root='data')

vs = VisualizeDataset(data_root='data')
vs.show_mean_width_heigth(preprocess=True)


train.start_training()