import random
import time

from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from suprivised_learning.model import create_model
import pandas as pd
from utils.grabscreen import windows_capture
import tensorflow as tf
import cv2

from utils.record_game_info import get_player_health, get_boss_health

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
model = create_model()
keys = pd.read_csv('../data/keys', names=['keys'])
keys = keys['keys'].values
choose = np.array([1, 0, 0, 0, 0, 0, 1, 0, 1, 0],dtype=np.bool)
a = keys[choose]
print(a)
# input()
option_names = keys
x_col = 'path'

seg_Frame = 4
input_shape = model.input.shape[1:3]
batch_size = 32
GAMMA = 0.9
epoch = 0
model.load_weights('model_classification.h5')

print(input_shape)
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    img = cv2.resize(img, (400,225))
    img = img / 255.
    img = img[None, ...,None]
    return img


def infence():
    count = 0
    screen = windows_capture('DARK SOULS III')
    player_health = get_player_health(screen)
    boos_health = get_boss_health(screen)

    x_t = preprocess_image(img=screen)

    s_t = np.concatenate([x_t] * 4, axis=-1)
    print(s_t.shape)
    # x_t = preprocess_image(screen)
    t = time.time()
    while True:
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        # a_t =
        if time.time()-t>8*200/100000:
            screen = windows_capture('DARK SOULS III')
            # player_health = get_player_health(screen)
            # boos_health = get_boss_health(screen)
            x_t1 = preprocess_image(img=screen)
            s_t = np.concatenate( [s_t[..., :3],x_t1], axis=3)

            res = model.predict(s_t)
            # print(res)
            # res =
            actions = keys[np.squeeze(res > 0.2)]
            print(actions)


            count += 1
            # break


infence()

# cv2.imshow('w',screen)
# cv2.waitKey(0)

#
#
# minibatch = []
# for _ in range(batch_size):
#     choose_index = random.randint(4, num - seg_Frame-1)
#     state_t = [img_to_array(load_img(df.iloc[i][x_col],
#                                      color_mode='grayscale', target_size=(256, 256)), ) for i in
#                range(choose_index - 4, choose_index)]
#     state_t = np.concatenate(state_t, axis=-1)
#     state_t = state_t / 255.
#     action_t = df.iloc[choose_index][option_names]
#     player_health_t = df.iloc[choose_index]['player_health']
#     boss_health_t = df.iloc[choose_index]['boss_health']
#
#     state_t1 = [img_to_array(load_img(df.iloc[i][x_col],
#                                       color_mode='grayscale', target_size=(256, 256)), ) for i in
#                 range(choose_index + seg_Frame - 4, choose_index + seg_Frame)]
#     state_t1 = np.concatenate(state_t1, axis=-1)
#     state_t1 = state_t1 / 255.
#     # label_t1 = df.iloc[choose_index + seg_Frame][option_names]
#     player_health_t1 = df.iloc[choose_index + seg_Frame]['player_health']
#     boss_health_t1 = df.iloc[choose_index + seg_Frame]['boss_health']
#
#     reward_t = -(boss_health_t1 - boss_health_t) + (player_health_t1 - player_health_t)
#     minibatch.append([state_t, action_t, reward_t, state_t1])
# state_t, action_t, reward_t, state_t1 = zip(*minibatch)
# state_t = np.array(state_t)
# state_t1 = np.array(state_t1)
# action_t = np.array(action_t)
#
# # targets = model.predict(state_t)
# targets = np.zeros(shape=(batch_size, 10))
# Q_sa = model.predict(state_t1)
# # print(Q_sa.shape)
# pos_num_repeats = np.sum(action_t == 1, axis=-1)
# posi_reward_t = np.repeat(reward_t, pos_num_repeats)
#
# targets[action_t == 1] = posi_reward_t + GAMMA * Q_sa[action_t == 1]
#
# num_repeats_neg = np.sum(action_t == 0, axis=-1)
# neg_reward_t = np.repeat(reward_t, num_repeats_neg)
#
# targets[action_t == 0] = -neg_reward_t + GAMMA * Q_sa[action_t == 0]
# loss = model.train_on_batch(state_t, targets)
# print('epoch{},loss:{}'.format(epoch,loss))
# epoch+=1
# if epoch%10==0:
#
#     model.save_weights('model.h5',)
