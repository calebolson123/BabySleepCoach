import mediapipe as mp
import sleepy_baby
import logging

from .helpers import check_eyes_open, check_mouth_open

class MediaAnalysis:
    """
     It analyzes frame provided on process_frame.

     Results are saved inside the object variable "analysis".
     It will be used later for decision logic to make the proper evaluation.
    """

    def __init__(self): 
        """__init__ Initialize Analyzer."""
        self.logger = logging.getLogger(self.__class__.__qualname__) 
        # TODO: try turning off refine_landmarks for performance, might not be needed
        self.face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1,
                                                    refine_landmarks=True,
                                                    min_detection_confidence=0.8,
                                                    min_tracking_confidence=0.8)
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.7,
                                           min_tracking_confidence=0.7)


    def process_baby_image_models(self, frame:sleepy_baby.frame.Frame):
        """
        process_baby_image_models analyze frame and get information.

        Results are stored in analysis variable inside object

        Parameters
        ----------
        frame : sleepy_baby.frame.Frame
            Get Frame object

        Returns
        -------
        _type_
            Returns dict with all findings and the objects for pose and face.
        """
        
        analysis = {
            "body_detected": False,
            "left_wrist_coords": None,
            "right_wrist_coords": None,
            "face_detected": False,
            "eyes_open": False,
            "mouth_open": False
        }
        results_pose = self.pose.process(frame)
        if results_pose.pose_landmarks:
            analysis["body_detected"] = True
            # 15 left-wrist, 16 right-wrist
            analysis["left_wrist_coords"] = (frame.shape[1] * results_pose.pose_landmarks.landmark[15].x, frame.shape[0] * results_pose.pose_landmarks.landmark[15].y)
            analysis["right_wrist_coords"] = (frame.shape[1] * results_pose.pose_landmarks.landmark[16].x, frame.shape[0] * results_pose.pose_landmarks.landmark[16].y)

            results = self.face.process(frame)
            if results.multi_face_landmarks:
                analysis["face_detected"] = True
                analysis["eyes_open"] = check_eyes_open(results.multi_face_landmarks[0].landmark)
                analysis["mouth_open"] = check_mouth_open(results.multi_face_landmarks[0].landmark)   
        else:
            analysis["body_found"] = False
        return analysis, results_pose.pose_landmarks, results.multi_face_landmarks