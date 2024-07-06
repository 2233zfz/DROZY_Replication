from drozy_dataloader import get_loader
import warp_norm
import torch
import cv2
import numpy as np
from model import gaze_network
import head_pose
import math

cam_drozy = './testpart/DROZY/kinect-intrinsics.yaml'  #drozy的相机参数
fs_drozy = cv2.FileStorage(cam_drozy, cv2.FILE_STORAGE_READ)
camera_matrix_drozy = fs_drozy.getNode('intrinsics').mat()
k, p = fs_drozy.getNode('k').mat(), fs_drozy.getNode('p').mat()
camera_distortion_drozy = np.zeros((5,1))
for i in range(3):
    camera_distortion_drozy[i]=k[i]
for j in range(2):
    camera_distortion_drozy[j+3]=p[j]
w_drozy = 512
h_drozy = 424
fs_drozy.release()

pre_trained_model_path = './ckpt/epoch_24_ckpt.pth.tar'
camera_matrix = camera_matrix_drozy
camera_distortion = camera_distortion_drozy
w = w_drozy
h = h_drozy
print('load the pre-trained model: ', pre_trained_model_path)
ckpt = torch.load(pre_trained_model_path, map_location=torch.device('cpu'))
model = gaze_network()
model.load_state_dict(ckpt['model_state'], strict=True)  # load the pre-trained model
model.eval()  # change it to the evaluation mode
predictor, face_detector = warp_norm.xmodel()

gc = np.array([0,0])
face_model_load = np.loadtxt('./modules/face_model.txt')
landmarks_eye_corner = [20,23,26,29]
face_model_eye_corner = face_model_load[landmarks_eye_corner, :]

# def faceRecognition():
#     data_path = get_loader()
#     image_path = image_path[0]
#     image = cv2.imread(image_path)
#     hr, ht, Ear, landmarks_sub, landmarks_of_all = warp_norm.xnorm(image, camera_matrix, camera_distortion,
#                                                                    predictor, face_detector)
#     # landmarks
#     eyelip_distance=[]
#     for i in range(2):
#         eyelip_distance.append((np.linalg.norm(landmarks[41 + 6 * i] - landmarks[37 + 6 * i], 2)) + np.linalg.norm(
#             landmarks[40 + 6 * i] - landmarks[38 + 6 * i], 2)/2)
#     eyelip_distance = np.mean(np.asarray(eyelip_distance))


def run():
    data_path = get_loader()
    counter = 0
    for image_path in data_path:
        image_path = image_path[0]
        image = cv2.imread(image_path)
        hr, ht, Ear, landmarks_sub, landmarks_of_all = warp_norm.xnorm(image, camera_matrix, camera_distortion,predictor, face_detector )
        ## landmarks
        # eyelip_distance=[]
        # for i in range(2):
        #     eyelip_distance.append((np.linalg.norm(landmarks[41 + 6 * i] - landmarks[37 + 6 * i], 2)) + np.linalg.norm(
        #         landmarks[40 + 6 * i] - landmarks[38 + 6 * i], 2)/2)
        # eyelip_distance = np.mean(np.asarray(eyelip_distance))
        # face_model
        Fc = warp_norm.xtrans_lip(face_model_eye_corner, hr, ht, w=w, h=h)
        distance_baseline = (np.linalg.norm(Fc[:, 0] - Fc[:, 1], ord=2, axis=0)+np.linalg.norm(Fc[:, 2] - Fc[:, 3], ord=2, axis=0))/2
        eyelip_distance,distance_landmarks_eye_corner = warp_norm.drozy_lip_distance(image)
        transfer_facter=distance_baseline/distance_landmarks_eye_corner
        real_eyelip_distance = eyelip_distance*transfer_facter
        # print(f'image_path:{image_path}')
        # print(f'distance_baseline:{distance_baseline}')
        # print(f'landmarks_eye_corner:{distance_landmarks_eye_corner}')
        # if(counter == 3):
        #     break;
            # cv2.namedWindow("Image")
            # cv2.imshow("Image", image)
            # key = cv2.waitKey(0)
            # if key == 27:  # 按下 ESC 键时，退出
            #     cv2.destroyAllWindows()
        with open('./test/drozy_3-1.txt','a',encoding='utf-8') as f:
            f.write(f'{counter}:     eye_lip_distance:{real_eyelip_distance:.4f}      Ear:{Ear:.4f}\n')
        counter = counter+1
    print('Done')


run()

