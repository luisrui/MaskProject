import cv2 as cv
import dataprocessor
from skimage.feature import hog
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics


def eye_dection(img):
    img = cv.GaussianBlur(img,(5,5),0) #高斯滤波
    equ = cv.equalizeHist(img) #直方图均衡化
    eyes_cascade = cv.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
    eyes_cascade.load("./haarcascades/haarcascade_eye_tree_eyeglasses.xml")  # 文件所在的具体位置
    '''此文件是opencv的haar眼睛特征分类器'''
    eyes = eyes_cascade.detectMultiScale(equ, scaleFactor = 1.1, minNeighbors = 3)          # 眼睛检测
    # for (x,y,w,h) in eyes:
    #     frame = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  # 画框标识眼部
    #     print("x y w h is",(x,y,w,h))
    #     # frame = cv.rectangle(img, (x, y+h), (x + 3*w, y + 3*h), (255, 0, 0), 2)  # 画框标识眼部
    return eyes

if __name__ == '__main__':
    dataset_mask = 'D:\\mask_task\\FaceMaskData_Resized'
    X_masked, y_masked = dataprocessor.LoadData(dataset_mask,0)
    X_masked = np.array(X_masked)

    # using hog to extract features
    X_eyes_features = []
    devide_length = 30
    cnt = 0
    y_eyes_labels = []
    for i in range(X_masked.shape[0]):
        for x in X_masked[i]:
            eyes_locations = eye_dection(x)
            # if len(eyes_locations) == 0:
            #     cv.imwrite(f'D://mask_task//error_pic//{cnt}.jpeg', x)
            #     cnt += 1
            # else:
            if len(eyes_locations) != 0:
                a, b, w, h = eyes_locations[0]
                x_eye = x[a : a + w, b : b + h] #crop the image into an eye and resize it into 50*50 pixels
                x_eye = cv.resize(x, dsize=(50,50),fx=1,fy=1,interpolation=cv.INTER_LINEAR)
                # random flip
                a = np.random.randint(0,3)
                if a < 1:
                    x_eye = cv.flip(x_eye, 0)
                elif 1 <= a < 2:
                    x_eye = cv.flip(x_eye, 1)
                elif 2 <= a < 3:
                    x_eye = cv.flip(x_eye, -1)
                eye_feature = hog(x_eye, orientations=8, pixels_per_cell=(10, 10), cells_per_block=(1, 1), visualize=False)
                t = 1 if i <= devide_length else 0
                y_eyes_labels.append(t)
                X_eyes_features.append(eye_feature)

    X_eyes_features = np.array(X_eyes_features)
    y_eyes_labels = np.array(y_eyes_labels)
    trainsamp = int(0.8*len(y_eyes_labels))
    testsamp = len(y_eyes_labels) - trainsamp
    clf = SVC()
    X_train, X_test, y_train, y_test = train_test_split(X_eyes_features, y_eyes_labels, train_size=trainsamp, test_size=testsamp, shuffle = True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    print(y_test)
    con_mtx = metrics.confusion_matrix(y_test, y_pred, labels = [1,0])
    print(f'accuracy={accuracy}')
    print(f'confusion_matrix={con_mtx}')
    # scores = cross_val_score(clf, X_eyes_features, y_eyes_labels, cv = 5, scoring='precision_weighted')
    # accuracy = scores.mean()
    # print(f'accuracy={accuracy}')

                

