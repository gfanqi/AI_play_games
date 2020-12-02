from PIL import ImageGrab
import sys

from PyQt5.QtWidgets import QApplication


import time
import win32gui, win32ui, win32con, win32api


def test_win32(count=50):
    i = 0
    begin = time.time()
    while True:
        i += 1
        hwnd = 0
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        MoniterDev = win32api.EnumDisplayMonitors(None, None)
        w = MoniterDev[0][2][2]
        h = MoniterDev[0][2][3]
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)
        saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)
        if i > count:
            break
    return time.time() - begin


def test_qt(count=50):
    app = QApplication(sys.argv)
    screen = QApplication.primaryScreen()
    i = 0
    begin = time.time()
    while True:
        i += 1
        screen.grabWindow(QApplication.desktop().winId())
        if i > count:
            break
    return time.time() - begin


def test_pil(count=50):
    i = 0
    begin = time.time()
    while True:
        i += 1
        ImageGrab.grab()
        if i > count:
            break
    return time.time() - begin




print("PIL:", test_pil(60))
print("PyQt5:", test_qt(60))
print("win32api:", test_win32(60))