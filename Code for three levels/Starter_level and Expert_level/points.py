import os
import cv2 as cv
import numpy as np
import dlib
import dataprocessor as dp


def imgshow(img):
    cv.imshow('sd',img)
    cv.waitKey(0)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

dataset_path_small = 'D:\\mask_task\\FaceData'
dataset_masked = 'D:\\mask_task\\FaceMaskData'
X_origin = [] 
for subject_name in os.listdir(dataset_path_small):
    subject_images_dir = os.path.join(dataset_path_small + '\\'+subject_name)
    temp_x_list = []
    for img_name in os.listdir(subject_images_dir):
        img = cv.imread(subject_images_dir+'\\'+img_name)
        # write code to read each 'img'
        temp_x_list.append(img)
        # add the img to temp_x_list
    X_origin.append(temp_x_list)

X_masked = [] 
for subject_name in os.listdir(dataset_masked):
    subject_images_dir = os.path.join(dataset_masked + '\\'+subject_name)
    temp_x_list = []
    for img_name in os.listdir(subject_images_dir):
        img = cv.imread(subject_images_dir+'\\'+img_name)
        # write code to read each 'img'
        temp_x_list.append(img)
        # add the img to temp_x_list
    X_masked.append(temp_x_list)

detector = dlib.get_frontal_face_detector()
# tmp = X
face_area = []# record each pic's face area's vertices in a list for cropping
for x_list in X_origin:
    temp_X_processed = []
    for x in x_list:
        # write the code to detect face in the image (x) using dlib facedetection library
        dets = detector(x,1)
        # write the code to crop the image (x) to keep only the face, resize the cropped image to 150x150
        d = dets[0]
        face_area.append(d)

predictor_path = 'D:\\mask_task\\shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
botlist = []
cnt = 0
fil = 50
inx = 0 # to index the sequence of face_area
target = [37,42,41,40,43,48,47,46,29]
for x_list, x_mask_list in zip(X_origin,X_masked):
    os.mkdir(f'D:\\mask_task\\UpperFace\\s{fil}')
    for x, xm in zip(x_list, x_mask_list):
        shape = predictor(x, face_area[inx])
        points = shape_to_np(shape)
        bot = 0
        for tar in target:
            bot = max(bot, points[tar-1][1])
        upper_face = xm[face_area[inx].top(): bot, face_area[inx].left():face_area[inx].right()]
        upper_face = cv.resize(upper_face,dsize=(150,150),fx=1,fy=1,interpolation=cv.INTER_LINEAR)
        cv.imwrite(f'D:\\mask_task\\UpperFace\\s{fil}\\{cnt + 1}.jpeg', upper_face)
        cnt = (cnt + 1) % 15
        inx += 1
    fil += 1
