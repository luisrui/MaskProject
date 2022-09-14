import cv2 as cv

def EyeDection(img, channel = 3):
    pro = cv.GaussianBlur(img,(5,5),0) #高斯滤波
    if channel == 3:
        pro = cv.cvtColor(pro, cv.COLOR_BGR2GRAY)
    equ = cv.equalizeHist(pro) #直方图均衡化
    eyes_cascade = cv.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
    eyes_cascade.load("D:\mask_task\haarcascades\haarcascade_eye_tree_eyeglasses.xml")  # 文件所在的具体位置
    eyes = eyes_cascade.detectMultiScale(equ, scaleFactor = 1.1, minNeighbors = 3)          # 眼睛检测
    return eyes

def LeftEyeDection(img, channel = 3):
    pro = cv.GaussianBlur(img,(5,5),0) #高斯滤波
    if channel == 3:
        pro = cv.cvtColor(pro, cv.COLOR_BGR2GRAY)
    equ = cv.equalizeHist(pro) #直方图均衡化
    lefteye_cascade = cv.CascadeClassifier("haarcascade_lefteye_2splits.xml")
    lefteye_cascade.load("D:\mask_task\haarcascades\haarcascade_lefteye_2splits.xml")  # 文件所在的具体位置
    eyes = lefteye_cascade.detectMultiScale(equ, scaleFactor = 1.1, minNeighbors = 3)          # 眼睛检测
    return eyes

def RightEyeDection(img, channel = 3):
    pro = cv.GaussianBlur(img,(5,5),0) #高斯滤波
    if channel == 3:
        pro = cv.cvtColor(pro, cv.COLOR_BGR2GRAY)
    equ = cv.equalizeHist(pro) #直方图均衡化
    righteye_cascade = cv.CascadeClassifier("haarcascade_righteye_2splits.xml")
    righteye_cascade.load("./haarcascades/haarcascade_righteye_2splits.xml")  # 文件所在的具体位置
    eyes = righteye_cascade.detectMultiScale(equ, scaleFactor = 1.1, minNeighbors = 3)          # 眼睛检测
    return eyes

def NoseDection(img, channel = 3):
    pro = cv.GaussianBlur(img,(5,5),0) #高斯滤波
    if channel == 3:
        pro = cv.cvtColor(pro, cv.COLOR_BGR2GRAY)
    equ = cv.equalizeHist(pro) #直方图均衡化
    nose_cascade = cv.CascadeClassifier("haarcascade_mcs_nose.xml")
    nose_cascade.load("D:\mask_task\haarcascades\haarcascade_mcs_nose.xml")  # 文件所在的具体位置
    nose = nose_cascade.detectMultiScale(equ, scaleFactor = 1.1, minNeighbors = 3)
    # print("nose is",(x,y,w,h))       
    # for (x,y,w,h) in nose:
    #     frame = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  # 画框标识眼部
    #     #print("nose is",(x,y,w,h))
    #     # frame = cv.rectangle(img, (x, y+h), (x + 3*w, y + 3*h), (255, 0, 0), 2)  # 画框标识眼部
    return nose