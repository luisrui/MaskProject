import os
import dataprocessor 
import cv2 as cv
import numpy as np
import face_recognition as fr
from sklearn.model_selection import train_test_split
from sklearn import metrics

def compare(imgA, imgB)->bool: #params: two openCV frame pics
    imgA_pro = imgA[:, :, ::-1]
    imgB_pro = imgB[:, :, ::-1]
    imgA_encoding = fr.face_encodings(imgA_pro)[0]
    imgB_encoding = fr.face_encodings(imgB_pro)[0]
    results = fr.compare_faces([imgA_encoding], imgB_encoding)
    return results[0]

dataset_origin = 'D:\\mask_task\\FaceData'
dataset_mask = 'D:\\mask_task\\FaceMaskData'

#import data: X_processed for gray pics of original faces, X_origin for gray pics of masked faces
X_origin , y_processed = dataprocessor.LoadData(dataset_origin)
#X_masked, y_masked = dataprocessor.LoadData(dataset_mask)
# X_processed = []
# for x_list in X_origin:
#     temp_x = []
#     for x in x_list:
#         x = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
#         temp_x.append(x)
#     X_processed.append(temp_x)


train_length = 30
test_length = int(input("input the test samples length:"))
#X_origin = dataprocessor.ShavePic_fr(X_origin)

#Initionalization of the train sample
test_samp_subsets = len(X_origin)
test_samp_pics = len(X_origin[0])
total_pics = test_samp_subsets * test_samp_pics

train_face_encoding = dataprocessor.Initialize(X_origin, train_length)

#Initialization of the test samples
flag = True

ARlabels = [1 for i in range(30*15)] + [0 for i in range(20*15)]
X_origin = [i for sub in X_origin for i in sub]
X_train, X_test, y_train, y_test = train_test_split(X_origin, ARlabels, train_size=total_pics - test_length, test_size=test_length, shuffle = True)

#The Train Process
y_pred = []
for i in range(test_length):
    flag = True
    unknown_image = X_test[i]
    unknown_image = unknown_image[:, :, ::-1]
    unknown_face_encoding = fr.face_encodings(unknown_image)[0]
    for j in range(train_length):
        results = fr.compare_faces([train_face_encoding[j]], unknown_face_encoding)
        if results[0] == True:
            y_pred.append(1)
            flag = False
            break
    if flag: y_pred.append(0)
accuracy = metrics.accuracy_score(y_test,y_pred)
print(f'accuracy={accuracy}')