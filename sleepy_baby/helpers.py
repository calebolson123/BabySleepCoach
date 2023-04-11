import numpy as np
import cv2
import os
from pyhatchbabyrest import PyHatchBabyRest
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()


def get_point_as_array(point):
    """
    get_point_as_array transforms landmarks coordinate in numpy array.

    Parameters
    ----------
    point : mediapipe.framework.formats.landmark_pb2.NormalizedLandmark
        Landmarks coordinate

    Returns
    -------
    numpy.array
        array of coordinates [x, y, z]
    """
    return np.array([point.x, point.y, point.z])

def get_distance_between_landmarks(landmark, ref0, ref1):
    p0 = get_point_as_array(landmark[ref0])
    p1 = get_point_as_array(landmark[ref1])
    return np.linalg.norm(p0 - p1)

def get_eyes_positions(landmarks): #TODO: comment it
    #Indexes of landmarks are reported at https://raw.githubusercontent.com/google/mediapipe/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    rh_right = get_point_as_array(landmarks[33])
    rh_left = get_point_as_array(landmarks[133])
    rv_top = get_point_as_array(landmarks[159])
    rv_bottom = get_point_as_array(landmarks[145])

    lh_right = get_point_as_array(landmarks[362])
    lh_left = get_point_as_array(landmarks[263])
    lv_top = get_point_as_array(landmarks[386])
    lv_bottom = get_point_as_array(landmarks[374])

    return {"Left Eye": {"right": lh_right, 
                         "left": lh_left, 
                         "top": lv_top, 
                         "bottom": lv_bottom},
            "Right Eye": {"right": rh_right, 
                          "left": rh_left, 
                          "top": rv_top, 
                          "bottom": rv_bottom}
            }

# Given x/y coords of eyes, returns a ratio representing "openness" of eyes
def closed_ratio(landmarks): #TODO: commen the function
    #Indexes of landmarks are reported at https://raw.githubusercontent.com/google/mediapipe/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    right_eye_width = get_distance_between_landmarks(landmarks, 33, 133)
    right_eye_height = get_distance_between_landmarks(landmarks, 159, 145)
    left_eye_width = get_distance_between_landmarks(landmarks, 362, 263)
    left_eye_height = get_distance_between_landmarks(landmarks, 386, 374)
    right_eye_ratio = right_eye_width / right_eye_height
    left_eye_ratio = left_eye_width / left_eye_height
    return (right_eye_ratio + left_eye_ratio) / 2

def check_eyes_open(landmarks, ratio_threshold=5): #TODO: comment it
    return closed_ratio(landmarks) <= ratio_threshold


def set_hatch(is_awake):
    print("attempting to boost hatch brightness")
    rest = PyHatchBabyRest(os.getenv('HATCH_IP'))
    rest.set_brightness(5)
    print("brightness: ", rest.brightness)

    return

def get_top_lip_height(landmarks):
    #Indexes of landmarks are reported at https://raw.githubusercontent.com/google/mediapipe/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    top_lip_left = get_distance_between_landmarks(landmarks, 39, 81)
    top_lip_center = get_distance_between_landmarks(landmarks, 0, 13)
    top_lip_right = get_distance_between_landmarks(landmarks, 269, 311)
    return  (top_lip_left + top_lip_center + top_lip_right) / 3

    
def get_bottom_lip_height(landmarks):
    #Indexes of landmarks are reported at https://raw.githubusercontent.com/google/mediapipe/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    bottom_lip_left = get_distance_between_landmarks(landmarks, 181, 178)
    bottom_lip_center = get_distance_between_landmarks(landmarks, 17, 14)
    bottom_lip_right = get_distance_between_landmarks(landmarks, 405, 402)
    return  (bottom_lip_left + bottom_lip_center + bottom_lip_right) / 3



def get_mouth_height(landmarks):
    #Indexes of landmarks are reported at https://raw.githubusercontent.com/google/mediapipe/a908d668c730da128dfa8d9f6bd25d519d006692/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    open_mouth_left = get_distance_between_landmarks(landmarks, 178, 81)
    open_mouth_center = get_distance_between_landmarks(landmarks, 14, 13)
    open_mouth_right = get_distance_between_landmarks(landmarks, 402, 311)
    return  (open_mouth_left + open_mouth_center + open_mouth_right) / 3


def check_mouth_open(landmarks, ratio = 0.8):
    top_lip_height =    get_top_lip_height(landmarks)
    bottom_lip_height = get_bottom_lip_height(landmarks)
    mouth_height =      get_mouth_height(landmarks)

    # if mouth is open more than lip height * ratio, return true.
    return mouth_height > min(top_lip_height, bottom_lip_height) * ratio


# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

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
    return cv2.resize(image, dim, interpolation=inter)

@lru_cache(maxsize=10)
def gamma_correction(gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    return np.array(table, np.uint8)