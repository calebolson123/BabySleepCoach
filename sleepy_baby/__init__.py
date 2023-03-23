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
import statistics
from dotenv import load_dotenv
# from cast_service import CastSoundService
from http.server import HTTPServer, SimpleHTTPRequestHandler
from .helpers import check_eyes_open, set_hatch, check_mouth_open, maintain_aspect_ratio_resize, gamma_correction

from .media_analysis import MediaAnalysis
from .decision_logic import DecisionLogic


class SleepyBaby():

    def __init__(self, frame_width, frame_height, decision_logic = DecisionLogic, debug=False):
        self.media = MediaAnalysis(frame_width, frame_height, debug)
        self.logic = decision_logic()

    def set_working_area(self, x_offset, y_offset, width, height):
        return self.media.set_working_area(x_offset, y_offset, width, height)
    
    def process_frame(self, frame):
        self.media.process_frame(frame)
        if self.media.analysis['body_detected']

    



class old:

    def __init__(self, x, y, width, height, debug=False):
        """
        __init__ _summary_

        Parameters
        ----------
        x : int, optional
            offset for crop image on x-axis, by default 700
        y : int, optional
            offset for crop image on y-axis, by default 125
        width : int, optional
            width of the interesting area, by default 800
        height : int, optional
            height of the interesting area, by default 1000
        debug : bool, optional
            show verbose log, by default False 
        """
        self.debug = debug
        self.logger = logging.getLogger(SleepyBaby.__name__)
        self.frame_dim = (1920,1080)
        self.next_frame = 0
        self.fps = 30
        self.x = x
        self.y = y
        self.h = height
        self.w = width
        self.shape = [height, width]
        self.mpPose = mp.solutions.pose
        self.mpFace = mp.solutions.face_mesh
        self.pose = self.mpPose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        # TODO: try turning off refine_landmarks for performance, might not be needed
        self.face = self.mpFace.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStyles = mp.solutions.drawing_styles

        self.eyes_open_q = deque(maxlen=30)
        self.awake_q = deque(maxlen=40)
        self.movement_q = deque(maxlen=40)
        self.eyes_open_state = False

        self.multi_face_landmarks = []
        self.is_awake = False
        self.ser = None # serial connection to arduino for controlling demon owl

        # If demon owl mode, setup connection to arduino and cast service for playing audio
        if os.getenv("OWL", 'False').lower() in ('true', '1'):
            print("\nCAWWWWWW\n")
            self.cast_service = CastSoundService()
            self.ser = serial.Serial('/dev/ttyACM0', 9600, timeout=0)

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

   

    def add_progress_bar_to_image(self, frame, percent): #TODO: move to class that manage decisions
        # draw progress bar
        bar_y_offset = 100
        bar_width = 200
        w = frame.shape[1]
        start_point = (int(w/2 - bar_width/2), 350 + bar_y_offset)
        end_point = (int(w/2 + bar_width/2), 370 + bar_y_offset)
        adj_percent = 1.0 if percent / .6 >= 1.0 else percent / .6
        progress_end_point = (int(w/2 - bar_width/2 + (bar_width*(adj_percent))), 370 + bar_y_offset)
        color = (255, 255, 117)
        progress_color = (0, 0, 255)
        thickness = -1
        frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
        frame = cv2.rectangle(frame, start_point, progress_end_point, progress_color, thickness)
        display_perc = int((percent * 100) / 0.6)
        display_perc = 100 if display_perc >= 100 else display_perc
        frame = cv2.putText(frame, str(display_perc) + "%", (int(w/2 - bar_width/2), 330 + bar_y_offset), 2, 1, (255,0,0), 2, 2)
        frame = cv2.putText(frame, "Awake", (int(w/2 - bar_width/2 + 85), 330 + bar_y_offset), 2, 1, (255,0,0), 2, 2)
        return frame



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
                