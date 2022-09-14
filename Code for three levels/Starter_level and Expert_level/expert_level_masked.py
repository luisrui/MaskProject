import os
import dataprocessor
import cv2 as cv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import face_recognition as fr
from sklearn import metrics


dataset_mask = 'D:\\mask_task\\FaceMaskData_Resized'

X_masked = []
y_masked = []
for subject_name in os.listdir(dataset_mask):
    subject_images_dir = os.path.join(dataset_mask + '\\'+subject_name)

    temp_x_list = []
    for img_name in os.listdir(subject_images_dir):
        img = fr.load_image_file(subject_images_dir+'\\'+img_name)
        # write code to read each 'img'
        temp_x_list.append(img)
        # add the img to temp_x_list
        y_masked.append(subject_name)
    X_masked.append(temp_x_list)

train_length = 30
test_length = int(input("input the test samples length:"))

# Initionalization of the train sample
test_samp_subsets = len(X_masked)
test_samp_pics = len(X_masked[0])
total_pics = test_samp_subsets * test_samp_pics

train_face_encoding = []
cnt = 0
for i in range(train_length):  # store the first pic's face encoding in each sub-sets
    tmp_face = X_masked[i][0]
    face_encodings = fr.face_encodings(
        tmp_face, known_face_locations=[(0, 150, 150, 0)])
    if len(face_encodings) == 0:
        cv.imwrite(f'D://mask_task//error_pic//{cnt}.jpeg', tmp_face)
        cnt += 1
    else:
        train_face_encoding.append(face_encodings[0])

# Initialization of the test samples
flag = True

ARlabels = [1 for i in range(30*15)] + [0 for i in range(20*15)]
X_masked = [i for sub in X_masked for i in sub]
X_train, X_test, y_train, y_test = train_test_split(
    X_masked, ARlabels, train_size=total_pics - test_length, test_size=test_length, shuffle=True)

# The Train Process
y_pred = []
for i in range(test_length):
    flag = True
    unknown_image = X_test[i]
    unknown_image = unknown_image[:, :, ::-1]
    unknown_face_encoding = fr.face_encodings(
        unknown_image, known_face_locations=[(0, 150, 150, 0)])[0]
    for j in range(train_length):
        results = fr.compare_faces(
            [train_face_encoding[j]], unknown_face_encoding)
        if results[0] == True:
            y_pred.append(1)
            flag = False
            break
    if flag:
        y_pred.append(0)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(y_pred)
print(f'accuracy={accuracy}')


# PCA Part

# # Initialize the masked data of openCV shape(BGR pic)
# dataset_mask = 'D:\\mask_task\\FaceMaskData_Resized'

# X_masked = []
# y_masked = []
# for subject_name in os.listdir(dataset_mask):
#     subject_images_dir = os.path.join(dataset_mask + '\\'+subject_name)

#     temp_x_list = []
#     for img_name in os.listdir(subject_images_dir):
#         img = cv.imread(subject_images_dir+'\\'+img_name, 0)
#         # write code to read each 'img'
#         temp_x_list.append(img)
#         # add the img to temp_x_list
#         y_masked.append(subject_name)
#     X_masked.append(temp_x_list)
# X_masked, y_masked = X_masked, y_masked

# # X_masked = dataprocessor.transfer_to_gray(X_masked)
# X_test = X_masked[0]

# X_masked_vec = []
# for pic in X_test:
#     pic_vec = np.reshape(pic, -1)
#     X_masked_vec.append(pic_vec)
# faces = np.array(X_masked_vec)
# faces = faces.T

# pca = PCA()
# pca.fit(np.transpose(faces))
# W = pca.components_
# m = pca.mean_

# plt.subplot(3,3,1)
# plt.imshow(m.reshape(150,150), cmap = 'gray')
# plt.title('mean')
# for i in range(3,11):
#     plt.subplot(3,3,i-1)
#     plt.imshow(W[i-3].reshape(150,150), cmap = 'gray')
#     t = i - 2
#     plt.title(f'eigenface{t}')
# plt.show()
