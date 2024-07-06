import h5py
import cv2
import warp_norm
import matplotlib
import sys
sys.path.append("./FaceAlignment")
import face_alignment
from skimage import io
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import gaze_network
from torchvision import transforms
import pickle
import gaze_normalize
import os
import math
import csv
from datetime import datetime,timedelta
import dlib
import time

def baselineBn(dataY,dataY_b):
    i=len(dataY)-1
    if i>0:
        Alpha = smoothingFactor(dataY_b,dataY, i)
        bI = (1 - Alpha) * dataY_b[i - 1] + Alpha * dataY[i]
        dataY_b.append(round(bI,4))
    return dataY_b

def smoothingFactor(dataY_b,dataY, n):
    a0 = 0.4
    ad = 15
    aa = 0.5
    ab = 2
    am = 0.7
    exp1 = (-1) * ad * ((dataY[n] - dataY[n - 1]) ** 2)
    exp_1 = math.exp(exp1)
    exp2 = (-1) * aa * (dataY[n] - dataY_b[n - 1]) if dataY[n] - dataY_b[n - 1] > 0 else 0
    exp_2 = math.exp(exp2)
    exp3 = (-1) * ab * (dataY_b[n - 1] - dataY[n]) if dataY_b[n - 1] - dataY[n] > 0 else 0
    exp_3 = math.exp(exp3)
    exp4 = dataY[n] - am * getMedian(dataY, n)
    exp_4 = 1 if exp4 >= 0 else 0
    return a0 * exp_1 * exp_2 * exp_3 * exp_4


def getMedian(dataY, n):
    d = []
    d.extend(dataY[1:n+1])
    d.sort()
    if len(d) % 2 == 0:
        return (d[len(d) // 2 - 1] + d[len(d) // 2]) / 2
    else:
        return d[len(d) // 2]

def frameBackandForth(data_normalized,blinkClassifier,flag):
    nextBegin=flag
    frameCount=1
    blinkClassifier[flag]=2
    for i in range(1,(flag+1 if flag<len(data_normalized)-flag else len(data_normalized)-flag)):
        if(data_normalized[flag-i]<=0.75):
            blinkClassifier[flag-i]=1
            frameCount+=1
            if(data_normalized[flag-i]<=0.65) and (data_normalized[flag-i+1]<=0.65):
                blinkClassifier[flag - i] = 2
        if(data_normalized[flag+i]<=0.75):
            blinkClassifier[flag + i] = 3
            frameCount += 1
            nextBegin+=1
            if (data_normalized[flag + i] <= 0.65) and (data_normalized[flag+i-1]<=0.65):
                blinkClassifier[flag + i] = 2
        if(data_normalized[flag-i]>0.75 and data_normalized[flag+i]>0.75):
            break
    return nextBegin,blinkClassifier,frameCount

def blinkDetecter(data_normalized):
    i = 0
    blinkClassifier=[0]*len(data_normalized)
    blinkCount=[0]*len(data_normalized)
    microsleepCount=[0]*len(data_normalized)
    while i < len(data_normalized):
        if (data_normalized[i] < 0.65):
            nextBegin, blinkClassifier,frameCount = frameBackandForth(data_normalized,blinkClassifier,i)
            microsleepCount[i]=frameCount
            blinkCount[i] = 1
            i = nextBegin
        i += 1
    return blinkClassifier,blinkCount,microsleepCount

startTime=time.time()

##获取相机参数
cam_drozy = r"D:\DROZY_and_NTHU\GazeNormalization-cpu_1\testpart\DROZY\kinect-intrinsics.yaml"  #drozy的相机参数
fs_drozy = cv2.FileStorage(cam_drozy, cv2.FILE_STORAGE_READ)
camera_matrix_drozy = fs_drozy.getNode('intrinsics').mat()
k, p = fs_drozy.getNode('k').mat(), fs_drozy.getNode('p').mat()
camera_distortion_drozy = np.zeros((5,1))
for i in range(3):
    camera_distortion_drozy[i]=k[i]
for j in range(2):
    camera_distortion_drozy[j+3]=p[j]
fs_drozy.release()

predictor = dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()

##读取模型
camera_matrix=camera_matrix_drozy
# trans = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ])
# model = gaze_network()
# model.cuda()
# pre_trained_model_path = r"D:\DROZY_and_NTHU\GazeNormalization-cpu_1\ckpt\epoch_24_ckpt.pth.tar"
# ckpt = torch.load(pre_trained_model_path)
# model.load_state_dict(ckpt['model_state'], strict=True)
# model.eval()

##视频处理
# video_path = r"D:\DROZY_and_NTHU\GazeNormalization-cpu_1\testpart\DROZY\videos_i8\11-2.mp4"
video_path=r"D:\DROZY_and_NTHU\geHang_GazeNormalization-main\testpart\webcam_face_2.avi"
cap = cv2.VideoCapture(video_path)
res = []
images = []
idx = 0
ret = True
camera_distortion=camera_distortion_drozy
preds = gaze_normalize.xmodel()
counter=0
dataX=[]
dataY=[]
dataY_b=[]
data_normalized = []
while counter<500:
    dataX.append(counter)
    counter+=1
    print(f'counter:{counter}')
    ret,image = cap.read()
    if ret == False:
        break
    gaze_normalize_eve = gaze_normalize.GazeNormalize(image, (0, 0), camera_matrix, camera_distortion,
                                                      predictor=predictor, is_video=True, image=image,
                                                      face_detector=face_detector)
    image_warp, real_eyelip_distance, Ear, R = gaze_normalize_eve.norm()
    if type(image_warp) == int:
        dataY.append(dataY[-1])
        dataY_b = baselineBn(dataY, dataY_b)
        continue
    dataY.append(real_eyelip_distance)
    if len(dataY) == 1:
        dataY_b.append(dataY[0])
    dataY_b = baselineBn(dataY, dataY_b)


for i in range(len(dataY)):
    opening = dataY[i] / dataY_b[i]
    data_normalized.append(round(opening, 4))

blinkClassifier, blinkCount, microsleepCount = blinkDetecter(data_normalized)

with open('./svm_data/txt_data/blinkClassifier_licheng','a+') as f:
    for k in range(len(blinkClassifier)):
        f.writelines(f'{blinkClassifier[k]}  {blinkCount[k]}  {microsleepCount[k]}  {data_normalized[k]}\n')
    print('txt文件已保存')
    f.close()

endTime=time.time()
print(f'eclapsed time:{endTime-startTime}')




