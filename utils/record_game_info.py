import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread('test1.jpg')

from matplotlib.backends.backend_agg import FigureCanvasAgg

from utils.grabscreen import windows_capture

from collections import Counter


def get_boss_health(img):
    H, W = img.shape[:2]
    # h_min = int(H * 0.833)
    # h_max = int(H * 0.845)
    w_min = int(W * 0.2908)
    w_max = int(W * 0.812)
    health = img[int(H * 0.8343), w_min:w_max, :3]

    health = np.logical_and(60 < health[..., 0], health[..., 0] < 98) * np.logical_and(60 < health[..., 1], health[
        ..., 1] < 98) * np.logical_and(
        90 < health[..., 2], health[..., 2] < 140)

    return health


def get_player_health(img):
    H, W = img.shape[:2]
    # h_min = int(H * 0.833)
    # h_max = int(H * 0.845)
    w_min = int(W * 0.1022) - 1
    # w_max = int(W * 0.352)
    w_max = w_min + 150
    health = img[int(H * (0.0765)), w_min:w_max, :3]  # 最后一个通道为R通道。血条是红色的
    # health_ = np.stack([health]*300,axis=0)
    health_ = img[int(H * (0.07)):int(H * (0.079)), w_min:w_max, :3]
    # health_ = cv2.resize(health_,dsize=None,fx=5,fy=10)
    # cv2.imshow('fd',health_)
    # cv2.waitKey(0)
    # print(np.min(health,axis=0))
    # print(np.max(health, axis=0))
    #
    # input()

    health = np.logical_and(25 < health[..., 0], health[..., 0] < 50) * np.logical_and(28< health[..., 1], health[
        ..., 1] < 50) * np.logical_and(
        70 < health[..., 2], health[..., 2] < 100)
    health = np.sum(health) / health.shape[0]
    return health


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    res = []
    last = 0
    img = cv2.imread(r'C:\Users\gfanqi\PycharmProjects\AI_play_games\data\supervised_learning\54549\image\eaa2ffa8-31a6-414c-aa2a-0e1c3c957456.jpg')
    # get_player_health(img)
    # for i in range(201, 1200):
    print(img.shape)
    i = 200
    new = get_player_health(img)
    print()
    print(new)
    # while True:
    #     i+=1
    # print('dark_souls/{}.jpg'.format(i))
    # img = windows_capture('DARK SOULS III')
    # new = get_boss_health(img)
    # print(new)
    # record=cv2.imread('tmp.png')
    # cv2.imshow('record',new)
    # cv2.waitKey(20)
    # print('\r{}'.format(new),end='')
    # res.append(new)
    # time.sleep(5)
    # if new > last+0.2:
    #     print(i)
    # last = new
    # plt.plot(res)
    # plt.show()
