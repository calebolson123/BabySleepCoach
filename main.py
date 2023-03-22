import cv2
from threading import Thread
import os
from collections import deque
import _thread
import logging
from dotenv import load_dotenv
# from cast_service import CastSoundService
from http.server import HTTPServer, SimpleHTTPRequestHandler

from sleepy_baby import SleepyBaby

# Uncomment if want phone notifications during daytime wakings.
# Configuration of telegram API key in this dir also needed.
# import telegram_send

#Set-up the logger
logfile = os.getenv("SLEEP_DATA_PATH") + '/sleepy_logs.log'
logging.basicConfig(filename=logfile,
                    filemode='a+',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

#Load configuration from .env file
load_dotenv()

# Queue shared between the frame publishing thread and the consuming thread
# This is to get around an underlying bug, described at end of this file.
frame_q = deque(maxlen=20)

#Load SleepyBaby
logging.info('Initializing...')
sleepy_baby = SleepyBaby()
logging.info('\nInitialization complete.')


# Below http server is used for the web app to request latest sleep data
class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super(CORSRequestHandler, self).end_headers()

def start_server():
    httpd = HTTPServer(('0.0.0.0', 8000), CORSRequestHandler)
    httpd.serve_forever()

_thread.start_new_thread(start_server, ())


def receive(producer_q):
    print("Start receiving frames.")
    cam_ip = os.environ['CAM_IP']
    cam_pw = os.environ['CAM_PW']
    connect_str = "rtsp://admin:" + cam_pw + "@" + cam_ip
    connect_str2 = connect_str + ":554" + "//h264Preview_01_main" # this might be different depending on camera used

    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp' # Use tcp instead of udp if stream is unstable
    c = cv2.VideoCapture(connect_str)

    next_frame = 0
    fps = 30
    while(c.isOpened()):
        ret, img = c.read()
        if ret:
            producer_q.append(img)


# Had to split frame receive and processing into different threads due to underlying FFMPEG issue. Read more here:
# https://stackoverflow.com/questions/49233433/opencv-read-errorh264-0x8f915e0-error-while-decoding-mb-53-20-bytestream
# Current solution is to insert into deque on the thread receiving images, and process on the other
p1 = Thread(target=receive, args=(frame_q,))
p2 = Thread(target=sleepy_baby.live, args=(frame_q,))
p1.start()
p2.start()

# Note: to test w/ recorded footage, comment out above threads, and uncomment next line
# TODO: use command line args rather than commenting out code
# sleepy_baby.recorded()