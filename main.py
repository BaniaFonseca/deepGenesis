from dataset.dataset import Dataset as ds

dataset = ds(train_dir='input/train/', test_dir='input/test/', width=500, height=500)

#dataset.show_mean_aspect_ratio()
#dataset.process_images()
#dataset.show_mean_aspect_ratio()

dataset.show_mean_width_heigth()
dataset.process_images()
dataset.show_mean_width_heigth()
