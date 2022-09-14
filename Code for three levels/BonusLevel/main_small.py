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
from sklearn.decomposition import PCA
from skimage.transform import integral_image
from skimage.feature import haar_like_feature

def imgshow(img):
    cv.imshow('sd',img)
    cv.waitKey(0)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)  
    return coords

datapath_upperface = 'D:\\mask_task\\UpperFace'
datapath_maskedface = 'D:\\mask_task\\FaceMaskData_Resized'
#load data of both upper face and masked face
X_upperface, y_upperface = dp.LoadData(datapath_upperface)
X_masked, y_masked = dp.LoadData(datapath_maskedface)
X_masked_p = []
for x_list in X_masked:
    x_mask_pro = []
    for x in x_list:
        t = x[0: (x.shape[1]//2-20), 0: x.shape[0]]
        p = cv.resize(t,dsize=(150,150),fx=1,fy=1,interpolation=cv.INTER_LINEAR)
        x_mask_pro.append(p)
    X_masked_p.append(x_mask_pro)
X_masked = X_masked_p

y_total = np.array(list(np.array([[i]*15 for i in range(50)]).flatten())*2)
X_total = X_upperface + X_masked
features = []
for x_list in X_total:
    for x in x_list:
        x = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
        x = cv.equalizeHist(x)
        left_eye_ii = integral_image(x)
        feature = haar_like_feature(left_eye_ii, 0, 0, left_eye_ii.shape[0], left_eye_ii.shape[1],'type-3-x')
        # feature = hog(x, orientations=8, pixels_per_cell=(10, 10), cells_per_block=(1, 1), visualize=False)
        features.append(feature)

pca = PCA(n_components = 300)
pca.fit(features)
W = pca.components_.T
m = pca.mean_

features = np.array(features)
features_for_train = features.dot(W)

X_train = features_for_train[0:750]
X_test = features_for_train[750:1500]
y_train = y_total[0:750]
y_test = y_total[750:1500]

#clf = SVC(C = 2,kernel = 'rbf')
clf = ensemble.AdaBoostClassifier(DecisionTreeClassifier(max_depth = 100), algorithm = "SAMME", n_estimators = 50)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test,y_pred)
print(f'accuracy={accuracy*100}%')

face_label = [i for i in range(50)]
con_mtx = metrics.confusion_matrix(y_test, y_pred, labels= face_label, sample_weight=None)
pd_con_mtx = pd.DataFrame(con_mtx)
# pd_con_mtx.to_excel(excel_writer='result.xlsx', sheet_name='sheet_1')