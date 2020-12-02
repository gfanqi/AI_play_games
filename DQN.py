import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('',img)
# 0.07222222222222222
# 0.13518518518518519
# 0.10104166666666667
# 0.3541666666666667


# cv2.waitKey(0)
H__, W__ = np.where(img == 255)
H, W = img.shape[:2]
print(np.min(H__) / H)
print(np.max(H__) / H)
print(np.min(W__) / W)
print(np.max(W__) / W)


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


def get_boos_health(img):
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
    w_min = int(W * 0.2908)
    w_max = int(W * 0.812)
    # rate = 0.9
    health = img[int(H * 0.8342), w_min:w_max, 2]
    # health = img[int(h_min * rate + h_max * (1 - rate)), w_min:w_max, 2]  # 最后一个通道为R通道。血条是红色的
    plt.plot(health)
    plt.show()
    # health = np.sum(health > 80) / health.shape[0]
    return health




print(get_boos_health(img))
#
# get_boos_health(img)
