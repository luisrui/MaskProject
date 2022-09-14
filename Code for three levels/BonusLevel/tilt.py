import os
import cv2 as cv
import numpy as np
import dlib
import dataprocessor as dp
import organ
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import math
from PIL import Image
from sklearn.decomposition import PCA


def imgshow(img):
    cv.imshow('sd', img)
    cv.waitKey(0)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
# w = cv.imread('D:\\mask_project\\FaceMaskData_Resized\\FaceMaskData_Resized\\s5\\3.jpeg')


def eye_detect(w):
    # wo = w.copy
    wx = np.array(w)
    eyes_det = cv.CascadeClassifier("haarcascade_eye.xml")
    eyes_det.load(f'D:\\mask_task\\haarcascades\\haarcascade_eye.xml')
    eyes = eyes_det.detectMultiScale(wx, scaleFactor=1.1, minNeighbors=3)
    # print(eyes)
    return eyes


# w = cv.resize(w, (900, 900), interpolation=cv.INTER_LINEAR)
# print(w.shape)
def face_tilt(w):
    wo = w.copy()
    eyes = eye_detect(wo)
    if(len(eyes) >= 2):
        x1 = eyes[0][0]+0.5*eyes[0][2]
        y1 = eyes[0][1]+0.5*eyes[0][3]
        x2 = eyes[1][0]+0.5*eyes[1][2]
        y2 = eyes[1][1]+0.5*eyes[1][3]
        # print('2eye')

        angle = math.degrees(math.atan2(y2-y1, x2-x1))
        # print(angle)
    elif(len(eyes) == 1):
        print('1eye')
        a, b, h, we = eyes[0]
        # wo = wo[0:75,25:125]
        wo = wo[b:b + we, a:a + h]
        wo = cv.cvtColor(wo, cv.COLOR_BGR2GRAY)
        wo = cv.equalizeHist(wo)
        wo = cv.GaussianBlur(wo, (3, 3), cv.BORDER_DEFAULT)
        wo = cv.Canny(wo, 90, 160)
        # cv.imshow('', wo)
        # cv.waitKey(0)
        wrench = np.where(wo == 255)
        wrench = np.array(wrench)
        wrench = wrench.T

        # print(wrench.shape)
        pca = PCA()
        pca.fit(wrench)
        # mean = pca.mean_
        mat = pca.components_
        # dist = np.dot(wrench, mat[1])
        # print(dist.shape)
        # idx = np.argmax(dist)
        # keypoint = wrench[idx]
        angle = math.degrees(math.atan2(mat[0][0], mat[0][1]))
        # print(angle)
        # if np.dot(keypoint-mean,mat[0])<0:
        # angle += 180
    elif(len(eyes) == 0):
        print('no eye')
        return w

    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180

    imgw = Image.fromarray(w)
    imgw = imgw.rotate(angle, fillcolor=127)
    pic = np.asarray(imgw)

    # print(pic.shape)
    # imgw.show()
    if abs(angle) < 55:
        return pic
    else:
        return w
