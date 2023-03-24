import logging
import numpy as np
import mediapipe as mp
import cv2

mp_utils = mp.solutions.drawing_utils

class Frame:
    def __init__(self, frame: cv2.Mat, x_offset: int =0, y_offset: int = 0, width:int = None, height:int= None):
        self.logger = logging.getLogger(self.__class__.__qualname__)
        self.frame = frame
        if (width is not None) and (height is not None):
            self.set_working_area(x_offset, y_offset, width, height)
        else:
            self.set_working_area(0,0, frame.shape[1], frame.shape[0])
        
    def set_working_area(self, x_offset: int, y_offset: int, width: int, height:int):
        """
        set_working_area will define a sub-area of frame to be analyze.

        This will help hardware to be faster and have a lower power consumption

        Parameters
        ----------
        x_offset : int
            offset for crop image on x-axis
        y_offset : int
            offset for crop image on y-axis
        width : int
            width of the interesting area
        height : int
            height of the interesting area
        """
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.height = height
        self.width = width
        self.clean_working_frame()

    def clean_working_frame(self) -> None:
        """Create a new working picture"""
        self.w_data = self.get_working_image()

    def get_working_image(self) -> np.ndarray:
        """
        Return the subset of the frame where analysis is done

        Returns
        -------
        np.ndarray
            Working area
        """
        return self.frame[self.y_offset:self.y_offset+self.height, self.x_offset:self.x_offset+self.width]
    
    def getAugmentedFrame(self) -> np.ndarray:
        """
        It generates the a new frame integrating the modified working area.

        Returns
        -------
        np.ndarray
            Image integrated
        """
        frame = self.frame.copy()
        frame[self.y_offset:self.y_offset+self.height, self.x_offset:self.x_offset+self.width] = self.w_data
        return frame
    

    def add_body_details(self, pose_landmarks):
        if pose_landmarks:
            self.logger.debug("Body detected in the frame")
            CUTOFF_THRESHOLD = 10  # head and face
            MY_CONNECTIONS = [t for t in mp.solutions.pose.POSE_CONNECTIONS if t[0] > CUTOFF_THRESHOLD and t[1] > CUTOFF_THRESHOLD]
            for id, lm in enumerate(pose_landmarks.landmark):
                if id <= CUTOFF_THRESHOLD:
                    lm.visibility = 0
                    continue
            mp_utils.draw_landmarks(self.w_data,
                                    pose_landmarks,
                                    MY_CONNECTIONS, 
                                    landmark_drawing_spec=mp_utils.DrawingSpec( color=(255, 0, 0),
                                                                                thickness=10,
                                                                                circle_radius=2),
                                    connection_drawing_spec=mp_utils.DrawingSpec(color=(0, 0, 255),
                                                                                thickness=5,
                                                                                circle_radius=2)
                                    )
            self.w_data = cv2.putText(self.w_data, "Left wrist", 
                                 (int(self.width * pose_landmarks.landmark[15].x),
                                  int(self.height * pose_landmarks.landmark[15].y)),
                                 2, 1, (255,0,0), 2, 2)
            self.w_data = cv2.putText(self.w_data, "Right wrist", 
                                 (int(self.width * pose_landmarks.landmark[16].x),
                                  int(self.height * pose_landmarks.landmark[16].y)),
                                  2, 1, (255,0,0), 2, 2)
        else:
            self.logger.debug("No body detected in frame")
    
    def add_analysis_frame(self):
        self.logger.debug("Draw the analysis frame")
        self.w_data = cv2.rectangle(self.w_data, [0,0], (self.w_data.shape[1], self.w_data.shape[0]), color=(0,255,0), thickness=5)

    def add_face_details(self, multi_face_landmarks):
        """
        add_face_details_to_image adds face details to image passed in arguments.

        Parameters
        ----------
        frame : numpy.ndarray
            starting image
        multi_face_landmarks : dict
            dictionary containing evaluation

        Returns
        -------
        numpy.ndarray
            image with some draws overlayed
        """
        if multi_face_landmarks:
            self.logger.debug("Face found in the frame")
            for face_landmarks in multi_face_landmarks:
                # INDICIES: https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
                # https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py

                mp.solutions.drawing_utils.draw_landmarks(
                    image=self.w_data,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_utils.DrawingSpec(color=(255, 150, 255), thickness=5, circle_radius=1))

                mp.solutions.drawing_utils.draw_landmarks(
                    image=self.w_data,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_utils.DrawingSpec(color=(255, 255, 0), thickness=5, circle_radius=1))
        else:
            self.logger.debug("No face found")