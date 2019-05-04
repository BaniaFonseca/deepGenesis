from train.train import Train
from util.visualize_dataset import VisualizeDataset

train = Train(data_root='data')
vds = VisualizeDataset(data_root='data')

vds.show_mean_aspect_ratio(preprocess=False)
