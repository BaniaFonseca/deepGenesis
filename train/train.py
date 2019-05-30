from dataset.dataset import Dataset
import tensorflow as tf

class Train(Dataset):

    def __init__(self, data_root):
        tf.compat.v1.enable_eager_execution
        super().__init__(data_root=data_root, image_height=256, image_width=256)



    def start_training(self):
        pass



