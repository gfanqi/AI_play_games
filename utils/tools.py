import sys

import numpy as np
import win32gui
from PyQt5.QtWidgets import QApplication


def windows_capture(name=None):
    hwnd = win32gui.FindWindow(None, name)
    a=QApplication(sys.argv)
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
    # pprint(get_hwnd_title())
    # print(dir(windows_capture()))
    import time
    t1 = time.time()
    for i in range(50):
        # image = windows_capture()
        arr = windows_capture('0')

        # cv2.imshow('fdsf',arr)
        # cv2.waitKey(0)
    print((time.time()-t1)/50)
    # # print(arr.shape)
    # # arr = cv2.cvtColor(arr,cv2.COLOR_BGRA2BGR)
    # # arr = arr[..., :-1]
    # print(arr.shape)
    # cv2.imshow('fds', arr)
    # cv2.waitKey()
    # # new_image = Image.fromarray(array)
    #
    # # for attr in dir(img):
    # #     try:
    # #         print(attr,img.__getattribute__(attr))
    # #         # img.__getattribute__(attr)
    # #     except:
    # #         pass
