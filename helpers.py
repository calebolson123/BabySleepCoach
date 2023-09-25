import math
import numpy as np
import cv2
import os
from pyhatchbabyrest import PyHatchBabyRest
from dotenv import load_dotenv

load_dotenv()

def euclidean(point, point1):
    x = point.x
    y = point.y
    x1 = point1.x
    y1 = point1.y

    return math.sqrt((x1 - x)**2 + (y1 - y)**2)


# Given x/y coords of eyes, returns a ratio representing "openness" of eyes
def closed_ratio(img, debug_img, landmarks, left_eye_indices, right_eye_indices):
    rh_right = landmarks[right_eye_indices[0]]
    rh_left = landmarks[right_eye_indices[8]]
    rv_top = landmarks[right_eye_indices[12]]
    rv_bottom = landmarks[right_eye_indices[4]]

    lh_right = landmarks[left_eye_indices[0]]
    lh_left = landmarks[left_eye_indices[8]]
    lv_top = landmarks[left_eye_indices[12]]
    lv_bottom = landmarks[left_eye_indices[4]]

    rhDistance = euclidean(rh_right, rh_left)
    rvDistance = euclidean(rv_top, rv_bottom)
    lvDistance = euclidean(lv_top, lv_bottom)
    lhDistance = euclidean(lh_right, lh_left)
    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance
    ratio = (reRatio + leRatio)/2

    # print('reRatio: ', reRatio)
    # print('leRatio: ', leRatio)
    return ratio


def set_hatch(is_awake):
    print("attempting to boost hatch brightness")
    rest = PyHatchBabyRest(os.getenv('HATCH_IP'))
    rest.set_brightness(5)
    print("brightness: ", rest.brightness)

    return


def check_eyes_open(landmarks, img, debug_img, left_eye_indices, right_eye_indices):
    eyes_closed_ratio = closed_ratio(img, debug_img, landmarks, left_eye_indices, right_eye_indices)
    ratio_threshold = 5
    if eyes_closed_ratio > ratio_threshold:
        return 0 # closed
    else:
        return 1 # open


def get_top_lip_height(landmarks):
    # 39 -> 81
    # 0 -> 13
    # 269 -> 311

    p39 = np.array([landmarks[39].x, landmarks[39].y, landmarks[39].z])
    p81 = np.array([landmarks[81].x, landmarks[81].y, landmarks[81].z])
    p0 = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    p13 = np.array([landmarks[13].x, landmarks[13].y, landmarks[13].z])
    p269 = np.array([landmarks[269].x, landmarks[269].y, landmarks[269].z])
    p311 = np.array([landmarks[311].x, landmarks[311].y, landmarks[311].z])

    d1 = np.linalg.norm(p39-p81)
    d2 = np.linalg.norm(p0-p13)
    d3 = np.linalg.norm(p269-p311)

    # print("average: ", (d1 + d2 + d3) / 3)
    return  (d1 + d2 + d3) / 3

    
def get_bottom_lip_height(landmarks):
    # 181 -> 178
    # 17 -> 14
    # 405 -> 402

    p181 = np.array([landmarks[181].x, landmarks[181].y, landmarks[181].z])
    p178 = np.array([landmarks[178].x, landmarks[178].y, landmarks[178].z])
    p17 = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
    p14 = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
    p405 = np.array([landmarks[405].x, landmarks[405].y, landmarks[405].z])
    p402 = np.array([landmarks[402].x, landmarks[402].y, landmarks[402].z])

    d1 = np.linalg.norm(p181-p178)
    d2 = np.linalg.norm(p17-p14)
    d3 = np.linalg.norm(p405-p402)

    # print("average: ", (d1 + d2 + d3) / 3)
    return  (d1 + d2 + d3) / 3


def get_mouth_height(landmarks):
    # 178 -> 81
    # 14 -> 13
    # 402 -> 311

    p178 = np.array([landmarks[178].x, landmarks[178].y, landmarks[178].z])
    p81 = np.array([landmarks[81].x, landmarks[81].y, landmarks[81].z])
    p14 = np.array([landmarks[14].x, landmarks[14].y, landmarks[14].z])
    p13 = np.array([landmarks[13].x, landmarks[13].y, landmarks[13].z])
    p402 = np.array([landmarks[402].x, landmarks[402].y, landmarks[402].z])
    p311 = np.array([landmarks[311].x, landmarks[311].y, landmarks[311].z])

    d1 = np.linalg.norm(p178-p81)
    d2 = np.linalg.norm(p14-p13)
    d3 = np.linalg.norm(p402-p311)

    # print("average: ", (d1 + d2 + d3) / 3)
    return  (d1 + d2 + d3) / 3


def check_mouth_open(landmarks):
    top_lip_height =    get_top_lip_height(landmarks)
    bottom_lip_height = get_bottom_lip_height(landmarks)
    mouth_height =      get_mouth_height(landmarks)

    # if mouth is open more than lip height * ratio, return true.
    ratio = 0.8
    if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
        return 1
    else:
        return 0


# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image, None

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the 0idth and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter), dim


def gamma_correction(og, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(og, table)