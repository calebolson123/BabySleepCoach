import cv2
import numpy as np
import time
from threading import Timer, Lock, Event, Thread
import os
import mediapipe as mp
from collections import deque
import _thread
import logging
import serial
import queue

from .frame import Frame
from .helpers import check_eyes_open, check_mouth_open
from .decision_logic import DecisionLogic




class SleepyBaby:
    """
     It analyzes frame provided on process_frame.

     Results are saved inside the object variable "analysis".
     It will be used later for decision logic to make the proper evaluation.
    """

    def __init__(self,
                 body_min_detection_confidence=0.8,
                 body_min_tracking_confidence=0.8,
                 face_min_detection_confidence=0.7,
                 face_min_tracking_confidence=0.7):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("SleepyBaby is starting")
        self.processed_frame = None #It is used to produce post-processed video
        self.process_t = None #Process Thread
        self.face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1,
                                                    refine_landmarks=True,
                                                    min_detection_confidence=body_min_detection_confidence,
                                                    min_tracking_confidence=body_min_tracking_confidence)
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=face_min_detection_confidence,
                                           min_tracking_confidence=face_min_tracking_confidence)
        self.set_working_area() #Set entire area as working area
        self.set_output() #Set default values
        self.logic = DecisionLogic()
        self.logger.info("SleepyBaby is configured")

    def set_output(self, 
                   show_frame=True, 
                   show_wrist_position=True, 
                   show_wrist_text=True, 
                   show_body_details=True, 
                   show_face_details=True,
                   show_progress_bar=True):
        self.show_frame = show_frame
        self.show_wrist_position = show_wrist_position
        self.show_wrist_text = show_wrist_text
        self.show_body_details = show_body_details
        self.show_face_details = show_face_details
        self.show_progress_bar = show_progress_bar

    def start_thread(self, frame_q, stop_event, pause=0.1, ):
        def process_loop(self, frame_q, stop_event, pause):
            while stop_event.is_set() is False:
                if len(frame_q)>0:
                    frame = self.processFrame(frame_q.pop(), return_image = self.processed_frame is None)
                    if frame is not None: 
                        self.processed_frame = frame
                time.sleep(pause)
        def evaluate_loop(logic, stop_event, pause=1):
            while stop_event.is_set() is False:
                logic.update()
                time.sleep(pause)
        self.process_t = Thread(target=process_loop, args=(self, frame_q, stop_event, pause))
        self.process_t.start()
        self.evaluate_t = Thread(target=evaluate_loop, args=(self.logic, stop_event))
        self.evaluate_t.start()



    def set_working_area(self, x_offset=0, y_offset=0, width=None, height=None):
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.width = width
        self.height = height
        self.working_area_inited = True

    def processFrame(self, image, return_image=True):
        frame = Frame(image, self.x_offset, self.y_offset, self.width, self.height)
        analysis, pose, face = self.process_baby_image_models(frame.w_data)
        self.logic.push(analysis)
        if return_image:
            if self.show_frame:
                frame.add_analysis_frame()
            if self.show_wrist_position or self.show_wrist_text:
                frame.add_wrist_position(pose, self.show_wrist_text)
            if self.show_body_details:
                frame.add_body_details(pose)
            if self.show_face_details:
                frame.add_face_details(face)
            if self.show_progress_bar:
                frame.add_progress_bar(self.logic.avg_awake)
            return frame.getAugmentedFrame()

    def process_baby_image_models(self, frame):
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
        results = None
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
        return analysis, results_pose.pose_landmarks, results.multi_face_landmarks if results is not None else None


class old:

    def __init__(self, x, y, width, height, debug=False):


        self.eyes_open_q = deque(maxlen=30)
        self.awake_q = deque(maxlen=40)
        self.movement_q = deque(maxlen=40)
        self.eyes_open_state = False

        self.is_awake = False
        self.ser = None # serial connection to arduino for controlling demon owl

        self.top_lip = frozenset([
            (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
            (82, 13), (13, 312), (312, 311), (311, 310),
            (310, 415), (415, 308),
            (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
            (37, 0), (0, 267),
            (267, 269), (269, 270), (270, 409), (409, 291),
        ])
        self.bottom_lip = frozenset([
            (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
            (17, 314), (314, 405), (405, 321), (321, 375),
            (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
            (14, 317), (317, 402), (402, 318), (318, 324),
        ]) 




    # This basically does the same thing as the live version, but is very useful for testing
    def recorded(self):
        cap = cv2.VideoCapture(os.getenv("VIDEO_PATH"))
        success, img = cap.read()
        while success:
            frame = None
            while frame is None:
                cur_time = time.time()
                if cur_time > self.next_frame:
                    frame = img
                    self.next_frame = max(
                        self.next_frame + 1.0 / self.fps, cur_time + 0.5 / self.fps
                    )

                    success, img = cap.read()

                    if all(e is not None for e in [frame, img]):
                        # bounds to actual run models/analysis on...no need to look for babies outside of the crib
                        x = 800
                        y = 250
                        h = 650
                        w = 600

                        if img.shape[0] > 1080 and img.shape[1] > 1920: # max res 1080p
                            img = maintain_aspect_ratio_resize(img, width=self.frame_dim[0], height=self.frame_dim[1])

                        img_to_process = img[y:y+h, x:x+w]

                        debug_img = self.frame_logic(img_to_process)

                        # reapply cropped and modified/marked up img back to img which is displayed
                        img[y:y+h, x:x+w] = debug_img

                        if os.getenv("DEBUG", 'False').lower() in ('true', '1'):
                            asleep = sum(self.awake_q) / len(self.awake_q) < 0.6
                            text = 'Sleepy Baby' if asleep else 'Wakey Baby'
                            text_color = (255,191,0) if asleep else (0,140,255)
                            cv2.putText(img, text, (int(img.shape[0]/2) + 250, int(img.shape[1]/2)), 2, 3, text_color, 2, 2)

                            cv2.rectangle(img=img, pt1=(x, y), pt2=(x+w, y+h), color=[153,50,204], thickness=2)
                            tmp = img[y:y+h, x:x+w]
                            img = gamma_correction(img, .4)

                            for face_landmarks in self.multi_face_landmarks:

                                    # INDICIES: https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts
                                    # https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py

                                    self.mpDraw.draw_landmarks(
                                        image=tmp,
                                        landmark_list=face_landmarks,
                                        connections=self.mpFace.FACEMESH_RIGHT_EYE,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=self.mpDraw.DrawingSpec(color=(255, 150, 255), thickness=1, circle_radius=1))
                                        # connection_drawing_spec=self.mpDrawStyles
                                        # .get_default_face_mesh_contours_style())
                                    self.mpDraw.draw_landmarks(
                                        image=tmp,
                                        landmark_list=face_landmarks,
                                        connections=self.mpFace.FACEMESH_LEFT_EYE,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=self.mpDraw.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1))
                                        # connection_drawing_spec=self.mpDrawStyles
                                        # .get_default_face_mesh_contours_style())

                                    self.mpDraw.draw_landmarks(
                                        image=tmp,
                                        landmark_list=face_landmarks,
                                        connections=self.top_lip,
                                        landmark_drawing_spec=None,#self.mpDraw.DrawingSpec(color=(255, 150, 255), thickness=2, circle_radius=2),
                                        connection_drawing_spec=self.mpDraw.DrawingSpec(color=(255, 150, 255), thickness=1, circle_radius=1))
                                    self.mpDraw.draw_landmarks(
                                        image=tmp,
                                        landmark_list=face_landmarks,
                                        connections=self.bottom_lip,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=self.mpDraw.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1))

                            img[y:y+h, x:x+w] = tmp

                            try:
                                img = cv2.resize(img, (960, 540))
                                cv2.imshow('baby', img)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                            except Exception as e:
                                print("Something went wrong: ", e)

   

    



    def process_image(self, frame):
        """
        This function will process image. 
        
        By default, only a interesting area defined by (x,y,h,w) is considered in processing.
        This avoids to waste resources to look for a baby outside crib

        Parameters
        ----------
        img : numpy.ndarray
            frame to be processed

        Returns
        -------
        numpy.ndarray
            image with overlays
        """

        #resize image if needed #FIXME: not working
        #if frame.shape[0] > self.frame_dim[0] and frame.shape[1] > self.frame_dim[1]: # max res 1080p
        #    frame = maintain_aspect_ratio_resize(frame, width=self.frame_dim[0], height=self.frame_dim[1])

        frame.flags.writeable = False #make the original frame read-only
        self.frame = frame
        working_area = cv2.cvtColor(frame[self.y:self.y+self.h, self.x:self.x+self.w], cv2.COLOR_BGR2RGB) #crop image and create a new image for processing
        
        analysis = self.process_baby_image_models(working_area)
        debug = frame.copy()
        debug = self.add_body_details_to_image(debug, analysis)
        debug = self.add_progress_bar_to_image(debug, 0.5)
        debug = self.add_face_details_to_image(debug, analysis)
        return debug
        
        #self.awake_voting_logic(debug_img)
        #self.movement_voting_logic(debug_img, body_found)
        #self.set_wakeness_status(debug_img)
        #self.periodic_wakeness_check()


    def live(self, consumer_q):
        img = None
        while True:
            if len(consumer_q) > 0:
                try:
                    img = consumer_q.pop() # consume image from queue
                except IndexError as e:
                    print('No images in queue: ', e)
                    continue

                img = self.process_image(img)
                cv2.imshow('baby', maintain_aspect_ratio_resize(img, width=self.frame_dim[0], height=self.frame_dim[1]))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                