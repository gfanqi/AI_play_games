import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread('test1.jpg')
import tqdm


def get_boss_health(img):
    H, W = img.shape[:2]
    # h_min = int(H * 0.833)
    # h_max = int(H * 0.845)
    w_min = int(W * 0.2908)
    w_max = int(W * 0.812)
    health = img[int(H * 0.8343), w_min:w_max, 2]
    health = np.sum(health > 100) / health.shape[0]
    return health


def get_player_health(img):
    H, W = img.shape[:2]
    # h_min = int(H * 0.833)
    # h_max = int(H * 0.845)
    w_min = int(W * 0.1022)-1
    # w_max = int(W * 0.352)
    w_max =w_min+159
    health = img[int(H * (0.0725)), w_min:w_max, 2]  # 最后一个通道为R通道。血条是红色的
    # health = health[:-1]-health[1:]
    # plt.plot(health)
    # plt.show()
    health = np.sum(health > 115) / health.shape[0]
    return health


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    res = []
    last = 0
    img = cv2.imread('dark_souls/1118.jpg')
    # get_player_health(img)
    for i in tqdm.tqdm(range(201, 1200)):
        img = cv2.imread('dark_souls/{}.jpg'.format(i))
        new = get_player_health(img)
        res.append(new)
        if new > last+0.2:
            print(i)
        last = new
    plt.plot(res)
    plt.show()
