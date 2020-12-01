import time
import win32gui, win32ui, win32con, win32api
import numpy as np

def window_capture(filename):
    hwnd = 0  # 窗口的编号，0号表示当前活跃窗口
    # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
    hwndDC = win32gui.GetWindowDC(hwnd)
    print(hwndDC)
    # 根据窗口的DC获取mfcDC
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    # mfcDC创建可兼容的DC
    saveDC = mfcDC.CreateCompatibleDC()
    # 创建bigmap准备保存图片
    saveBitMap = win32ui.CreateBitmap()
    # 获取监控器信息
    MoniterDev = win32api.EnumDisplayMonitors(None, None)
    w = MoniterDev[0][2][2]
    h = MoniterDev[0][2][3]
    # print w,h　　　#图片大小
    # 为bitmap开辟空间
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    # 高度saveDC，将截图保存到saveBitmap中
    saveDC.SelectObject(saveBitMap)
    # 截取从左上角（0，0）长宽为（w，h）的图片
    saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)
    # print(saveDC)
    # saveBitMap.SaveBitmapFile(saveDC, filename)


# beg = time.time()
# for i in range(10):
#     window_capture("haha.jpg")
# end = time.time()
# print(end - beg)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import *
import win32gui
import sys

hwnd = win32gui.FindWindow(None, 'AI_play_games – tools.py')
print(hwnd)
app = QApplication(sys.argv)
screen = QApplication.primaryScreen()
img = screen.grabWindow(hwnd).toImage()
# print(img.text())

print(np.array(img))
# print(np.array(img))

# print(type(img))
a = img.save("screenshot.jpg")
print(a)
# hwnd_title = {}
#
#
# def get_all_hwnd(hwnd, mouse):
#     if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
#         hwnd_title.update({
#             hwnd: win32gui.GetWindowText(hwnd)
#         })
#
#
# win32gui.EnumWindows(get_all_hwnd, 0)
# for h, t in hwnd_title.items():
#     if t != '':
#         print(h, t)
#         pass
#     # else:
#         # print(t)
