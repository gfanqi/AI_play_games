import random

from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from suprivised_learning.model import create_model
import pandas as pd

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
model = create_model()
data = pd.read_csv('../data/supervised_learning/72255/modifyed_label.csv')
# ImageDataGenerator
# load_img
# Frame_fps

keys = pd.read_csv('../data/keys', names=['keys'])
keys = keys['keys'].tolist()

option_names = keys
x_col = 'path'
df = data
seg_Frame = 4
num = len(df)
batch_size = 32
GAMMA = 0.9
epoch =0
model.load_weights('model_classification.h5')
while True:
    minibatch = []
    for _ in range(batch_size):
        choose_index = random.randint(4, num - seg_Frame-1)
        state_t = [img_to_array(load_img(df.iloc[i][x_col],
                                         color_mode='grayscale', target_size=(256, 256)), ) for i in
                   range(choose_index - 4, choose_index)]
        state_t = np.concatenate(state_t, axis=-1)
        state_t = state_t / 255.
        action_t = df.iloc[choose_index][option_names]
        player_health_t = df.iloc[choose_index]['player_health']
        boss_health_t = df.iloc[choose_index]['boss_health']

        state_t1 = [img_to_array(load_img(df.iloc[i][x_col],
                                          color_mode='grayscale', target_size=(256, 256)), ) for i in
                    range(choose_index + seg_Frame - 4, choose_index + seg_Frame)]
        state_t1 = np.concatenate(state_t1, axis=-1)
        state_t1 = state_t1 / 255.
        # label_t1 = df.iloc[choose_index + seg_Frame][option_names]
        player_health_t1 = df.iloc[choose_index + seg_Frame]['player_health']
        boss_health_t1 = df.iloc[choose_index + seg_Frame]['boss_health']

        reward_t = -(boss_health_t1 - boss_health_t) + (player_health_t1 - player_health_t)

        minibatch.append([state_t, action_t, reward_t, state_t1])
    state_t, action_t, reward_t, state_t1 = zip(*minibatch)
    state_t = np.array(state_t)
    state_t1 = np.array(state_t1)
    action_t = np.array(action_t)

    # targets = model.predict(state_t)
    targets = np.zeros(shape=(batch_size, 10))
    Q_sa = model.predict(state_t1)
    # print(Q_sa.shape)
    pos_num_repeats = np.sum(action_t == 1, axis=-1)
    posi_reward_t = np.repeat(reward_t, pos_num_repeats)

    targets[action_t == 1] = posi_reward_t + GAMMA * Q_sa[action_t == 1]

    num_repeats_neg = np.sum(action_t == 0, axis=-1)
    neg_reward_t = np.repeat(reward_t, num_repeats_neg)

    targets[action_t == 0] = -neg_reward_t + GAMMA * Q_sa[action_t == 0]
    loss = model.train_on_batch(state_t, targets)
    print('epoch{},loss:{}'.format(epoch,loss))
    epoch+=1
    if epoch%10==0:

        model.save_weights('model.h5',)