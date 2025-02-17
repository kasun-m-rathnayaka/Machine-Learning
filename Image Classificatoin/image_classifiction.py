import os

import numpy as np
from skimage.io import imread
from skimage.transform import resize

# prepare data
input_dir = './data/'
categoris = ['empty', 'not_empty']

data = []
labels = []

for category_index, category in enumerate(categoris):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15,))
        data.append(img.flatted())
        labels.append(category_index)

data = np.asarray(data)
labels = np.asarray(labels)

# train / test split


#  train classifier

# test performance