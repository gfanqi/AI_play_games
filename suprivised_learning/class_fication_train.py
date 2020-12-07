import os
import random
from pprint import pprint

from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from suprivised_learning.model import create_model
import pandas as pd

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
model = create_model()
# pd.read_csv('../data/supervised_learning/54549/')
# print(os.listdir('../data/supervised_learning/54549/'))
data = pd.read_csv('../data/supervised_learning/modifyed_label.csv', index_col=[0])

selected_data = data[['a', 'r', 'spacebar', 'tab', 'right_mouse_button',
                      'left_mouse_button', 'd', 'w', 'q', 's', ]]
print(np.sum(selected_data))

print(data.columns)

#
keys = pd.read_csv('../data/keys', names=['keys'])
keys = keys['keys'].tolist()
input_size = model.input.shape[1:3]

index_dict = {}
for key in keys:
    indexes = (data.index[data[key] == 1])
    indexes = indexes[indexes > 40]
    index_dict[key] = indexes.tolist()
    # index_dict[key]
pprint(index_dict)

# exit()
print(input_size)
option_names = keys
x_col = 'path'
df = data
seg_Frame = 4
num = len(df)
# batch_size = len(keys)*3
GAMMA = 0.9
epoch = 0
model.load_weights('model_classification.h5', by_name=True, skip_mismatch=True)
while True:
    minibatch = []
    for key in keys:
        index_list = index_dict[key]

        # if choose_index

        while True:
            flag = False
            choose_indexes = random.sample(index_list, 20)
            count = 0
            for choose_index in choose_indexes:
                state_t = [img_to_array(load_img(df.iloc[i * 8 + choose_index][x_col],
                                                 color_mode='grayscale', target_size=input_size), ) for i in
                           range(-3, 1)]
                t = df.iloc[choose_index]['time'] - df.iloc[-3 * 8 + choose_index]['time']
                if t > 10000:
                    flag = True
                    continue
                state_t = np.concatenate(state_t, axis=-1)
                state_t = state_t / 255.
                action_t = df.iloc[choose_index][option_names]
                player_health_t = df.iloc[choose_index]['player_health']
                boss_health_t = df.iloc[choose_index]['boss_health']

                minibatch.append([state_t, action_t, ])
                count += 1
                if count == 3: break
                # minibatch.append(state_t)
            if count == 3:break
            if flag: continue
    print(len(minibatch))
    random.shuffle(minibatch)
    state_t, action_t, = zip(*minibatch)
    state_t = np.array(state_t)
    action_t = np.array(action_t, dtype=np.float32)

    # targets = model.predict(state_t)
    loss = model.train_on_batch(state_t, action_t)

    print('epoch{},loss:{}'.format(epoch, loss))
    epoch += 1
    if epoch % 10 == 0:
        model.save_weights('model_classification.h5', )
