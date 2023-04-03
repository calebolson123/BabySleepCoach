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
        return self.frame[self.y_offset:self.y_offset+self.height, self.x_offset:self.x_offset+self.width].copy()
    
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
    

    def add_body_details(self, 
                         pose_landmarks, 
                         landmark_color=(255, 150, 255), 
                         landmark_thickness=2, 
                         landmark_circle_radius=2):
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
                                    landmark_drawing_spec=mp_utils.DrawingSpec( color=landmark_color,
                                                                                thickness=landmark_thickness,
                                                                                circle_radius=landmark_circle_radius)
                                    )
        else:
            self.logger.debug("No body detected in frame")
    
    def add_wrist_position(self, pose_landmarks, show_text=True, text_color=(255,0,0), point_color=(255,0,0), point_radius=2):
        if pose_landmarks:
            left_wrist = (int(self.width * pose_landmarks.landmark[15].x), int(self.height * pose_landmarks.landmark[15].y))
            right_wrist = (int(self.width * pose_landmarks.landmark[16].x), int(self.height * pose_landmarks.landmark[16].y))

            cv2.circle(self.w_data, left_wrist, radius=point_radius, color=point_color, thickness=-1)
            cv2.circle(self.w_data, right_wrist, radius=point_radius, color=point_color, thickness=-1)
            if show_text:
                self.w_data = cv2.putText(self.w_data, "Left wrist", left_wrist, 2, 1, text_color, 2, 2)
                self.w_data = cv2.putText(self.w_data, "Right wrist", right_wrist, 2, 1, text_color, 2, 2)

    def add_analysis_frame(self):
        self.logger.debug("Draw the analysis frame")
        self.w_data = cv2.rectangle(self.w_data, [0,0], (self.w_data.shape[1], self.w_data.shape[0]), color=(0,255,0), thickness=5)

    def add_face_details(self, 
                         multi_face_landmarks, 
                         details_thickness = 1,
                         left_eye_color=(255, 255, 0), 
                         right_eye_color=(255, 150, 255),
                         top_lip_color = (255, 150, 255),
                         bottom_lip_color = (255, 255, 0)
                         ):
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

                if right_eye_color is not None:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=self.w_data,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_utils.DrawingSpec(color=right_eye_color, thickness=details_thickness, circle_radius=1))

                if left_eye_color is not None:
                    mp.solutions.drawing_utils.draw_landmarks( #a[0:9]+a[20:29]
                        image=self.w_data,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_utils.DrawingSpec(color=left_eye_color, thickness=details_thickness, circle_radius=1))
                
                if top_lip_color is not None:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=self.w_data,
                        landmark_list=face_landmarks,
                        connections=list(map(lambda x: x[29:40]+x[9:20], [list(mp.solutions.face_mesh.FACEMESH_LIPS)]))[0],
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_utils.DrawingSpec(color=top_lip_color, thickness=details_thickness, circle_radius=1))
                
                if bottom_lip_color is not None:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=self.w_data,
                        landmark_list=face_landmarks,
                        connections=list(map(lambda x: x[0:9]+x[20:29], [list(mp.solutions.face_mesh.FACEMESH_LIPS)]))[0],
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_utils.DrawingSpec(color=bottom_lip_color, thickness=details_thickness, circle_radius=1))
        else:
            self.logger.debug("No face found")

    def add_progress_bar(self, percent, bar_width=500, bar_height = 20, bar_y_offset=100, backcolor=(255, 255, 117), forecolor=(0,0,255), textcolor=(255,0,0)): 
        # draw progress bar
        adj_percent = min(1.0, percent / 0.6)
        start_point = (int(self.width/2 - bar_width/2), self.height - bar_y_offset)
        end_point = (start_point[0] + bar_width, start_point[1] + bar_height)
        mid_point = (start_point[0] + int(bar_width * adj_percent), start_point[1] + bar_height)
        text_y_position = start_point[1] - int(bar_height / 5)

        self.w_data = cv2.rectangle(self.w_data, start_point, end_point, backcolor, thickness = -1)
        self.w_data = cv2.rectangle(self.w_data, start_point, mid_point, forecolor, thickness = -1)
        self.w_data = cv2.putText(self.w_data, str(int(adj_percent * 100)) + "% Awake", (start_point[0], text_y_position), 2, 1, textcolor, 2, 2)