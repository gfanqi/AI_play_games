import shutil
import sys
from pprint import pprint

import numpy as np
import win32gui
from PyQt5.QtWidgets import QApplication


def windows_capture(name=None):
    hwnd = win32gui.FindWindow(None, name)
    a = QApplication(sys.argv)
    screen = QApplication.primaryScreen()
    img = screen.grabWindow(hwnd).toImage()
    size = img.size()
    s = img.bits().asstring(size.width() * size.height() * img.depth() // 8)  # format 0xffRRGGBB
    arr = np.fromstring(s, dtype=np.uint8).reshape((size.height(), size.width(), img.depth() // 8))
    return arr


def get_hwnd_title():
    hwnd_title = {}

    def get_all_hwnd(hwnd, mouse):
        if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
            hwnd_title.update({
                hwnd: win32gui.GetWindowText(hwnd)
            })

    win32gui.EnumWindows(get_all_hwnd, 0)
    return hwnd_title


if __name__ == '__main__':
    import cv2
    import time
    import os

    if os.path.exists('../dark_souls'):shutil.rmtree('../dark_souls')
    os.mkdir('../dark_souls')
    # while True:
    FPS = 20
    SecPerFPS = 1 / FPS
    t1= t0 = time.time()
    i = 0
    while True:
        if time.time() - t1 > SecPerFPS*0.995:
            t1 = time.time()
            img = windows_capture('DARK SOULS III')
            # img = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
            # print(img.shape)
            cv2.imwrite('dark_souls/{}.jpg'.format(i),img[...,:3])
            i += 1
        # if i == 120: break
    # print(time.time() - t0)
