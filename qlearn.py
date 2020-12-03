#!/usr/bin/env python
from __future__ import print_function

import argparse
import json
# import wrapped_flappy_bird as game
import random
from collections import deque

import numpy as np
import skimage as skimage
from skimage import transform, color, exposure
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam



GAME = 'bird'  # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 8  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVATION = 200  # timesteps to observe before training
EXPLORE = 3000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.05 # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

img_rows, img_cols = 80, 80
# Convert image into Black and white
img_channels = 4  # We stack 4 frames
# info = "\r {prifix} TIMESTEP {t}/ STATE {state}/ EPSILON {epsilon}/ ACTION {action_index}/ REWARD {r_t}/ Q_MAX {Q_sa}/ LOSS {loss}/"
# print('\r' + prifix + "TIMESTEP", t, "/ STATE", state,
#       "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,
#       "/ Q_MAX ", np.max(Q_sa), "/ Loss ", loss, end='')

def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(8, 8,), strides=(4, 4), padding='same',
                            input_shape=(img_rows, img_cols, img_channels)))  # 80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    print("We finish building the model")
    return model


def trainNetwork(model, args):
    # open up a game state to communicate with emulator


    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t, (80, 80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))

    x_t = x_t / 255.0

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    # print (s_t.shape)

    # In tensorflow.keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4

    if args['mode'] == 'Run':
        OBSERVE = 999999999  # We keep observe, never train
        epsilon = FINAL_EPSILON
        print("Now we load weight")
        model.load_weights("trained_model/model.h5", by_name=True)
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        print("Weight load successfully")
    else:  # We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

        model.load_weights("model.h5")
    t = 0
    while True:
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        prifix = ''
        # choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                prifix = "------Random Action-------"
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                prifix = "-----model prediction-----"
                q = model.predict(s_t)  # input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1

        # We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1 / 255.0

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # Now we do the experience replay
            state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
            state_t = np.concatenate(state_t)
            state_t1 = np.concatenate(state_t1)
            targets = model.predict(state_t)
            Q_sa = model.predict(state_t1)
            print(targets)
            print(targets[range(BATCH), action_t])
            print(action_t)
            targets[range(BATCH), action_t] = reward_t + GAMMA * np.max(Q_sa, axis=1) * np.invert(terminal)
            # print(targets)
            input()
            loss += model.train_on_batch(state_t, targets)

        s_t = s_t1
        t = t + 1

        # save progress every 10000 iterations
        if t % 1000 == 0:
            print("\nNow we save model")
            model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print(f"\r {prifix} TIMESTEP {t}/ STATE {state}/ EPSILON {epsilon}/"
              f" ACTION {action_index}/ REWARD {r_t}/ Q_MAX {np.max(Q_sa)}/ LOSS {loss}/",end='')


def playGame(args):
    model = buildmodel()
    trainNetwork(model, args)


def main():
    args = {'mode': 'Train'}
    playGame(args)


if __name__ == "__main__":
    main()
