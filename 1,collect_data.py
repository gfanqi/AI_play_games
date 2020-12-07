import random
import time
from pprint import pprint

import cv2
import uuid
from utils.getkeys import key_check
from utils.grabscreen import get_hwnd_title, windows_capture
from utils.record_game_info import get_boss_health, get_player_health
import pandas as pd
import os

# pprint()
data = pd.DataFrame()
data_dir = 'data/supervised_learning/{}'.format(random.randint(0, 100000))
image_dir = os.path.join(data_dir, 'image')
label_dir = os.path.join(data_dir, 'label')
if not os.path.exists(image_dir): os.makedirs(image_dir)
if not os.path.exists(label_dir): os.makedirs(label_dir)

start = False
last_keys = ''
t0 = time.time()
while True:
    # get_hwnd_title()
    img = windows_capture('DARK SOULS III')
    # time.sleep(0.1)
    # print(img)
    image_name = uuid.uuid4().__str__() + '.jpg'
    # boss_health = get_boss_health(img)
    t = (time.time() - t0) * 10000
    player_health = get_player_health(img)
    boss_health = get_boss_health(img)
    keys = key_check()
    record_keys = keys
    if len(keys) > 0 and 'q' in keys and 'left_mouse_button' in keys:
        if (last_keys == keys):
            record_keys = []
    last_keys = keys

    # print(keys)
    if 'p' in keys:
        keys = []
        start = True
        print('开始')
    if 'esc' in keys:
        start = False
        keys = []
        print('暂停')
        # break
    if start:
        res = pd.DataFrame({
            'image_name': image_name,
            'keys': record_keys.__str__(),
            'player_health': player_health,
            'boss_health': boss_health,
            'time': t,
        }, index=[0])
        res.to_csv(os.path.join(label_dir, uuid.uuid4().__str__() + '.csv'))
        cv2.imwrite(os.path.join(image_dir, image_name), img[..., :3])
