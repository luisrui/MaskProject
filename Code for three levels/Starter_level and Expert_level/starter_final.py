import os
import cv2 as cv
import dlib
import numpy as np
from sklearn import metrics
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# To print the image using openCV directly
def printImg(img):
    cv.imshow('ds',img) 
    cv.waitKey(0)
# To transfer the shape parts into a list
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)  
    return coords

# T1  start _______________________________________________________________________________
# Read in Dataset

dataset_path = "D:\\mask_task\\FaceData"

X = [] 
y = []
for subject_name in os.listdir(dataset_path):
    subject_images_dir = os.path.join(dataset_path + '\\'+subject_name)
    
    temp_x_list = []
    for img_name in os.listdir(subject_images_dir):
        img = cv.imread(subject_images_dir+'\\'+img_name)
        # write code to read each 'img'
        temp_x_list.append(img)
        # add the img to temp_x_list
        y.append(subject_name)
    X.append(temp_x_list)
    # add the temp_x_list to X

# T1 end ____________________________________________________________________________________

# T2 start __________________________________________________________________________________
# Preprocessing
edge = 20
detector = dlib.get_frontal_face_detector()
# tmp = X
face_area = []# record each pic's face area's vertices in a list for cropping
X_processed = []
for x_list in X:
    temp_X_processed = []
    for x in x_list:
        # write the code to detect face in the image (x) using dlib facedetection library
        dets = detector(x,1)
        # write the code to crop the image (x) to keep only the face, resize the cropped image to 150x150
        d = dets[0]
        face_area.append(d)
        # write the code to convert the image (x) to grayscale
        # x = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
        # append the converted image into temp_X_processed
        temp_X_processed.append(x)
    # # append temp_X_processed into  X_processed
    X_processed.append(temp_X_processed)

# T2 end ____________________________________________________________________________________

#Store the image of BGR color into a set called FaceMaskData


# T3 start __________________________________________________________________________________
# Create masked face dataset

X_masked = []
X_pro_cropped = []
predictor_path = 'D:\\mask_task\\shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
cnt = 0
fil = 0
inx = 0 # to index the sequence of face_area

for x_list in X_processed:
    temp_x_masked = []
    temp_p_cropped = []
    os.mkdir(f'D:\\mask_task\\FaceMaskData\\s{fil}')
    for x in x_list:
        # write the code to detect face in the image (x) using dlib facedetection library
        shape = predictor(x, face_area[inx])
        # write the code to add synthetic mask as shown in the project problem description
        points = shape_to_np(shape)
        bowl_shape = np.array([points[i] for i in range(1, 16)])
        bowl_shape = np.append(bowl_shape, points[30]).reshape(16, 2)
        triangle = np.array([points[28], points[48], points[54]])
        tmp = x.copy()
        ds = cv.fillPoly(tmp, [bowl_shape], (255, 255, 255))
        tmp = cv.fillPoly(ds, [triangle], (255, 255, 255))
        # resize the pic in processed and the pic in masked to 150*150 for training
        # x_crop = x.copy()
        # x_crop = x_crop[face_area[inx].top():face_area[inx].bottom(),face_area[inx].left():face_area[inx].right()]
        # x_crop = cv.resize(x_crop,dsize=(150,150),fx=1,fy=1,interpolation=cv.INTER_LINEAR)
        # tmp = tmp[face_area[inx].top():face_area[inx].bottom(),face_area[inx].left():face_area[inx].right()]
        # tmp = cv.resize(tmp,dsize=(150,150),fx=1,fy=1,interpolation=cv.INTER_LINEAR)
        cv.imwrite(f'D:\\mask_task\\FaceMaskData\\s{fil}\\{cnt + 1}.jpeg',tmp)
        cnt = (cnt + 1) % 15
        inx += 1
        # temp_p_cropped.append(x_crop)
        # append the converted image into temp_X_masked
        temp_x_masked.append(tmp)
    # append temp_X_masked into  X_masked
    X_masked.append(temp_x_masked)
    X_pro_cropped.append(temp_p_cropped)
    fil += 1

# T3 end ____________________________________________________________________________________


# # T4 start __________________________________________________________________________________
# # Build a detector that can detect presence of facemask given an input image
# X_processed = X_pro_cropped
# X_features = []
# clf = SVC()
# X_total = X_masked+X_processed
# # print(len(X_total))
# # print(len(y))
# y_total = [1 for i in range(len(y))]+[0 for i in range(len(y))]
# trainsamp = int(0.8*len(y_total))
# testsamp= int(len(y_total)-trainsamp)
# for x_list in X_total:
#     temp_X_features = []
#     for x in x_list:
#         x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10), cells_per_block=(1, 1), visualize=False)
#         temp_X_features.append(x_feature)
#     X_features.append(temp_X_features)
# # write code to split the dataset into train-set and test-set
# # flatten the 3-dimension features into 2 dimensions
# X_features = [i for k in X_features for i in k]
# X_features = np.array(X_features)
# y_total = np.array(y_total)
#     # print(y_total.shape)
#     # print(X_features.shape)
# X_train, X_test, y_train, y_test = train_test_split(X_features, y_total, train_size=trainsamp, test_size=testsamp, shuffle = True)
# #print(y_test)
#     # print(f'Xtrain = {X_train.shape} Xtest = {X_test.shape}')
#     # print(f'ytrain = {y_train.shape} ytest = {y_test.shape}')
# print(f'y_test = {y_test}')
# # write code to train and test the SVM classifier as the facemask presence detector

# # clf.fit(X_train, y_train)
# # y_pred = clf.predict(X_test)
# # accuracy = metrics.accuracy_score(y_test,y_pred)

# #Using cross validation
# scores = cross_val_score(clf, X_features, y_total, cv = 5, scoring='precision_weighted')
# accuracy = scores.mean()
# print(f'accuracy={accuracy}')