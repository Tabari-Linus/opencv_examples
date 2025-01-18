import numpy as np
import cv2

def get_limits(color):
    c = np.unit8([[color]]) # here insert thebgr value which you want to convert to hsv
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    lowerlimit = hsvC[0][0][0] - 10, 100, 100
    upperlimit = hsvC[0][0][0] + 10, 255, 255

    lowerlimit = np.array(lowerlimit, dtype=np.unit8)
    upperlimit = np.array(upperlimit, dtype=np.unit8)

    return lowerlimit, upperlimit