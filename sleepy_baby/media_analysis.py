import numpy as np
import cv2
import mediapipe as mp
import logging

from .helpers import check_eyes_open, check_mouth_open

class MediaAnalysis:

    def __init__(self, frame_width, frame_height, debug=False):
        """
        __init__ Initialize Analyzer.

        Parameters
        ----------
        debug : bool, optional
            show verbose log, by default False 
        """
        self.debug = debug
        self.logger = logging.getLogger(MediaAnalysis.__name__)
        self.set_working_area(0, 0, frame_width, frame_height)

        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        # TODO: try turning off refine_landmarks for performance, might not be needed
        self.face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.8, min_tracking_confidence=0.8)

        self.pose_landmark = None
        self.multi_face_landmarks = None

    def set_working_area(self, x, y, width, height):
        """
        set_working_area will define a sub-area of frame to be analyze.

        This will help hardware to be faster and have a lower power consumption.

        Parameters
        ----------
        x : int
            offset for crop image on x-axis
        y : int
            offset for crop image on y-axis
        width : int
            width of the interesting area
        height : int
            height of the interesting area
        """
        self.x = x
        self.y = y
        self.h = height
        self.w = width
        self.shape = [height, width]

    def process_baby_image_models(self, img):
        """
        process_baby_image_models analyze frame and get information.

        Parameters
        ----------
        img : numpy.ndarray
            Image to be processed. It is already cropped

        Returns
        -------
        dict
            analysis dict that contains all findings of the image
        """

        analysis = {
            "body_detected": False,
            "left_wrist_coords": None,
            "right_wrist_coords": None,
            "face_detected": True,
            "eyes_open": False,
            "mouth_open": False
        }

        results_pose = self.pose.process(img)
        if results_pose.pose_landmarks:
            analysis["body_detected"] = True
            self.pose_landmarks = results_pose.pose_landmarks
            # 15 left-wrist, 16 right-wrist
            analysis["left_wrist_coords"] = (self.shape[1] * self.pose_landmarks.landmark[15].x, self.shape[0] * self.pose_landmarks.landmark[15].y)
            analysis["right_wrist_coords"] = (self.shape[1] * self.pose_landmarks.landmark[16].x, self.shape[0] * self.pose_landmarks.landmark[16].y)

            results = self.face.process(img)
            if results.multi_face_landmarks:
                self.multi_face_landmarks = results.multi_face_landmarks
                analysis["face_detected"] = True
                analysis["eyes_open"] = check_eyes_open(results.multi_face_landmarks[0].landmark)
                analysis["mouth_open"] = check_mouth_open(results.multi_face_landmarks[0].landmark)   
        else:
            self.pose_landmark = None
            self.multi_face_landmarks = None
            analysis["body_found"] = False
        return analysis

    def add_body_details_to_image(self, frame, analysis):
        w_area = frame[self.y:self.y+self.h, self.x:self.x+self.w]
        cv2.rectangle(w_area, [0, 0], self.shape, color=(0, 255, 0), thickness=5) #Draw Analysis Area
        #Draw body lines
        if analysis['body_detected']:
            CUTOFF_THRESHOLD = 10  # head and face
            MY_CONNECTIONS = [t for t in mp.solutions.pose.POSE_CONNECTIONS if t[0] > CUTOFF_THRESHOLD and t[1] > CUTOFF_THRESHOLD]
            for id, lm in enumerate(self.pose_landmarks.landmark):
                if id <= CUTOFF_THRESHOLD:
                    lm.visibility = 0
                    continue
            mp.solutions.drawing_utils.draw_landmarks(w_area, 
                                              self.pose_landmarks, 
                                              MY_CONNECTIONS, 
                                              landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), 
                                                                                                          thickness=10, 
                                                                                                           circle_radius=2),
                                              connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), 
                                                                                                          thickness=5, 
                                                                                                           circle_radius=2)
                                             )
            w_area = cv2.putText(w_area, "Left wrist", (int(analysis["left_wrist_coords"][0]), int(analysis["left_wrist_coords"][1])), 2, 1, (255,0,0), 2, 2)
            w_area = cv2.putText(w_area, "Right wrist", (int(analysis["right_wrist_coords"][0]), int(analysis["right_wrist_coords"][1])), 2, 1, (255,0,0), 2, 2)
            frame[self.y:self.y+self.h, self.x:self.x+self.w] = w_area
        return frame

    def add_face_details_to_image(self, frame, analysis):
        """
        add_face_details_to_image adds face details to image passed in arguments

        Parameters
        ----------
        frame : numpy.ndarray
            starting image
        analysis : dict
            dictionary containing evaluation

        Returns
        -------
        numpy.ndarray
            image with some draws overlayed
        """
        if analysis['face_detected']:
            self.logger.info(f"Face Detected. Eyes are {'open' if analysis['eyes_open'] else 'close'} and mouth is {'open' if analysis['mouth_open'] else 'close'}")
            w_area = frame[self.y:self.y+self.h, self.x:self.x+self.w]
            for face_landmarks in self.multi_face_landmarks:
                # INDICIES: https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
                # https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py

                mp.solutions.drawing_utils.draw_landmarks(
                    image=w_area,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 150, 255), thickness=5, circle_radius=1))
                    # connection_drawing_spec=self.mpDrawStyles
                    # .get_default_face_mesh_contours_style())

                mp.solutions.drawing_utils.draw_landmarks(
                    image=w_area,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 0), thickness=5, circle_radius=1))
                    # connection_drawing_spec=self.mpDrawStyles
                    # .get_default_face_mesh_contours_style())
            frame[self.y:self.y+self.h, self.x:self.x+self.w] = w_area
        else:
            self.logger.info("Face is not detected")
        return frame

    def add_body_details_to_image(self, frame, analysis):
        """
        add_body_details_to_image adds body details to image passed in arguments

        Parameters
        ----------
        frame : numpy.ndarray
            starting image
        analysis : dict
            dictionary containing evaluation

        Returns
        -------
        numpy.ndarray
            image with some draws overlayed
        """
        w_area = frame[self.y:self.y+self.h, self.x:self.x+self.w]
        cv2.rectangle(w_area, [0, 0], self.shape, color=(0, 255, 0), thickness=5) #Draw Analysis Area
        #Draw body lines
        if analysis['body_detected']:
            CUTOFF_THRESHOLD = 10  # head and face
            MY_CONNECTIONS = [t for t in mp.solutions.pose.POSE_CONNECTIONS if t[0] > CUTOFF_THRESHOLD and t[1] > CUTOFF_THRESHOLD]
            for id, lm in enumerate(self.pose_landmarks.landmark):
                if id <= CUTOFF_THRESHOLD:
                    lm.visibility = 0
                    continue
            mp.solutions.drawing_utils.draw_landmarks(w_area, 
                                              self.pose_landmarks, 
                                              MY_CONNECTIONS, 
                                              landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), 
                                                                                                          thickness=10, 
                                                                                                           circle_radius=2),
                                              connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), 
                                                                                                          thickness=5, 
                                                                                                           circle_radius=2)
                                             )
            w_area = cv2.putText(w_area, "Left wrist", (int(analysis["left_wrist_coords"][0]), int(analysis["left_wrist_coords"][1])), 2, 1, (255,0,0), 2, 2)
            w_area = cv2.putText(w_area, "Right wrist", (int(analysis["right_wrist_coords"][0]), int(analysis["right_wrist_coords"][1])), 2, 1, (255,0,0), 2, 2)
            frame[self.y:self.y+self.h, self.x:self.x+self.w] = w_area
            return frame
