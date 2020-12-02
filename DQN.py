import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test1.jpg')


# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('',img)
# 0.07222222222222222
# 0.13518518518518519
# 0.10104166666666667
# 0.3541666666666667


# cv2.waitKey(0)


# print(H__/H)
# print(W__/W)
# a = np.where(img)
# print(a)
# a = img[int(H * 0.833):int(H * 0.845), int(W * 0.2908):int(W * 0.812), :]
# # a = img[int(H * 0.833):int(H * 0.845), int(W*0):int(W * 1), :]
# # a = img[int(H * ):int(H * 0.845), int(W * 0.2908):int(W * 0.812), :]
# # plt.imshow(a)
# print(W * 0.2908)
# print(W * 0.812)
# # a = cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
# print(a.shape)
#
# # a = a[4:5,0:100]
# a = np.mean(a[..., 2], axis=0)
# plt.plot(a)
# plt.show()


def get_boss_health(img):
    H, W = img.shape[:2]
    # h_min = int(H * 0.833)
    # h_max = int(H * 0.845)
    w_min = int(W * 0.2908)
    w_max = int(W * 0.812)
    # rate = 0.9
    health = img[int(H * 0.8342), w_min:w_max, 2]
    # health = img[int(h_min * rate + h_max * (1 - rate)), w_min:w_max, 2]  # 最后一个通道为R通道。血条是红色的
    plt.plot(health)
    plt.show()
    # health = np.sum(health > 80) / health.shape[0]
    return health


def get_player_health(img):
    H, W = img.shape[:2]
    # h_min = int(H * 0.833)
    # h_max = int(H * 0.845)
    w_min = int(W * 0.1022)
    w_max = int(W * 0.352)
    # rate = 0.9
    # health = img[int(H *0.15), w_min:w_max, 2]
    # img = img[int(H*0.07):int(H*0.085),w_min:w_max,:]
    # cv2.imshow('ds',img)
    # cv2.waitKey(0)
    # for i in range(10):
    #     print(i*0.001+0.07)
    health = img[int(H * (0.0725)), w_min:w_max, 2]  # 最后一个通道为R通道。血条是红色的
        # plt.plot(health)
        # plt.show()
        # input()
    health = np.sum(health > 120) / health.shape[0]
    return health


# print(get_boos_health(img))
print(get_player_health(img))
# get_boos_health(img)
