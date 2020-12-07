import os
from functools import reduce

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import cv2

data = pd.read_csv('./supervised_learning/72255/label.csv', index_col=[0])
data['keys'] = data['keys'].map(lambda x: eval(x))
# print(data.columns)

# print(keys)
keys = pd.read_csv('../data/keys', names=['keys'])
# print(preds)
keys = keys['keys'].tolist()

for key in set(keys):
    data[key] = data['keys'].map(lambda x:int(key in x))
# data.to_csv()
# print()

data['path'] = data['image_name'].map(lambda x:os.path.abspath(os.path.join('../data/supervised_learning/72255/image',x)))

path_list = data.path.tolist()
#
for index,path in enumerate(path_list):
    if index <66:continue
    img = cv2.imread(path)
    cv2.imshow('windows',img)
    cv2.waitKey(0)
    print('\r{}'.format(index))
# data.to_csv('./supervised_learning/72255/modifyed_label.csv')