import os
import cv2 as cv
import numpy as np
import face_recognition as fr
import dlib

def LoadData(dataset_path, channel = 1) -> list:#import data into a list
    X = [] 
    y = []
    for subject_name in os.listdir(dataset_path):
        subject_images_dir = os.path.join(dataset_path + '\\'+subject_name)
        
        temp_x_list = []
        for img_name in os.listdir(subject_images_dir):
            img = cv.imread(subject_images_dir+'\\'+img_name, channel)
            # write code to read each 'img'
            temp_x_list.append(img)
            # add the img to temp_x_list
            y.append(subject_name)
        X.append(temp_x_list)
    return X, y

def ShavePic_fr(X_masked) -> list:#recognize the faces location and resize the pics into 150*150 pixels
    X = []
    for x_list in X_masked:
        temp_x = []
        for x in x_list:
            t = x[:, :, ::-1] #transfer to fr framework
            dets = fr.face_locations(t)
            top, right, bottom, left = dets[0]
            t = t[top : bottom, left : right]
            t = cv.resize(t, dsize=(150,150), fx=1, fy=1, interpolation=cv.INTER_LINEAR)
            temp_x.append(t)
        X.append(temp_x)
    return X

def Initialize(dataset, train_length = 30) -> list: #initialize the first pic in each subset for encoding
    train_face_encoding = []
    cnt = 0
    for i in range(train_length):#store the first pic's face encoding in each sub-sets
        tmp_face = dataset[i][0]
        tmp_face_RGB = tmp_face[:, :, ::-1] #transfer openCV pic to fr pic
        face_encodings = fr.face_encodings(tmp_face_RGB)
        if len(face_encodings) == 0:
            cv.imwrite(f'D://mask_task//error_pic//{cnt}.jpeg', tmp_face)
            cnt += 1
        else:
            train_face_encoding.append(face_encodings[0])
    return train_face_encoding

def transfer_to_gray(dataset):
    tmp = []
    for x_list in dataset:
        tmp_x = []
        for x in x_list:
            tmp.append(cv.cvtColor(x,cv.COLOR_BGR2GRAY))
        tmp.append(tmp_x)
    return tmp

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)  
    return coords

def wear_mask(img):
    detector = dlib.get_frontal_face_detector()
    predictor_path = 'D:\\mask_task\\shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)
    face_locations = detector(img,1)
    if len(face_locations) == 0:
        return False    #if not recognize the face, return false
    else:
        shape = predictor(img, face_locations[0])
        points = shape_to_np(shape)
        bowl_shape = np.array([points[i] for i in range(1, 16)])
        bowl_shape = np.append(bowl_shape, points[30]).reshape(16, 2)
        triangle = np.array([points[28], points[48], points[54]])
        tmp = img.copy()
        ds = cv.fillPoly(tmp, [bowl_shape], (255, 255, 255))
        tmp = cv.fillPoly(ds, [triangle], (255, 255, 255))
        return tmp