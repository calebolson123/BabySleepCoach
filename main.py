import cv2
from threading import Thread, Event
import os
from collections import deque
import _thread
import logging
import time
from dotenv import dotenv_values
# from cast_service import CastSoundService
from http.server import HTTPServer, SimpleHTTPRequestHandler

from sleepy_baby import SleepyBaby
from sleepy_baby import Frame

# Uncomment if want phone notifications during daytime wakings.
# Configuration of telegram API key in this dir also needed.
# import telegram_send

#Load configuration from .env file
config = dotenv_values()

#Config command ling
parser = argparse.ArgumentParser(
    prog="Sleepy Baby",
    description="Library to monitor the status of your baby based on image recognition"
)

parser.add_argument('-s', '--source', type=str, default=config['VIDEO_PATH'], help="Input path for video")
parser.add_argument('-v', '--verbose', action="store_true", default=config['DEBUG'].lower()=="true", help="Activate Debug Mode")
parser.add_argument('--log-on-screen', action="store_true", help="Show logs on screen instead of saving on file")
parser.add_argument('--log-path', default=config['SLEEP_DATA_PATH'], help="Set log path")
parser.add_argument('-r', '--recorded', action="store_true", help="Input is a recorded video. delay should be applied to simulate real-time")

args = parser.parse_args()



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
#frame_q = deque(maxlen=20)

#Load SleepyBaby
#logging.info('Initializing...')
#sleepy_baby = SleepyBaby()
#logging.info('\nInitialization complete.')


# Below http server is used for the web app to request latest sleep data
#class CORSRequestHandler(SimpleHTTPRequestHandler):
#    def end_headers(self):
#        self.send_header('Access-Control-Allow-Origin', '*')
#        self.send_header('Access-Control-Allow-Methods', 'GET')
#        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
#        return super(CORSRequestHandler, self).end_headers()

#def start_server():
#    httpd = HTTPServer(('0.0.0.0', 8000), CORSRequestHandler)
#    httpd.serve_forever()

#_thread.start_new_thread(start_server, ())


#def receive(producer_q):
#    print("Start receiving frames.")
#    cam_ip = os.environ['CAM_IP']
#    cam_pw = os.environ['CAM_PW']
#    connect_str = "rtsp://admin:" + cam_pw + "@" + cam_ip
#    connect_str2 = connect_str + ":554" + "//h264Preview_01_main" # this might be different depending on camera used

 #   os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp' # Use tcp instead of udp if stream is unstable
#    c = cv2.VideoCapture(connect_str)

#    next_frame = 0
#    fps = 30
#    while(c.isOpened()):
#        ret, img = c.read()
#        if ret:
#            producer_q.append(img)


# Had to split frame receive and processing into different threads due to underlying FFMPEG issue. Read more here:
# https://stackoverflow.com/questions/49233433/opencv-read-errorh264-0x8f915e0-error-while-decoding-mb-53-20-bytestream
# Current solution is to insert into deque on the thread receiving images, and process on the other
#p1 = Thread(target=receive, args=(frame_q,))
#p2 = Thread(target=sleepy_baby.live, args=(frame_q,))
#p1.start()
#p2.start()

# Note: to test w/ recorded footage, comment out above threads, and uncomment next line
# TODO: use command line args rather than commenting out code
# sleepy_baby.recorded()

frame_q = deque(maxlen=2)
terminate_event = Event()

#Load SleepyBaby
sleepy_baby = SleepyBaby(body_min_detection_confidence=0.1, body_min_tracking_confidence=0.1)
sleepy_baby.set_working_area(800, 300, 1100, 550)
sleepy_baby.set_output(show_body_details=True)
sleepy_baby.start_thread(frame_q, terminate_event)

def show_video(sb_obj):
    while terminate_event.is_set() is False:
        if sb_obj.processed_frame is not None:
            cv2.imshow("VIDEO", cv2.resize(sb_obj.processed_frame, (960,540)))
            cv2.waitKey(1)
            sb_obj.processed_frame = None
            time.sleep(0.3)
p2 = Thread(target=show_video, args=(sleepy_baby,))
p2.start()

try:
    vcap = cv2.VideoCapture("rtsp://192.168.62.198/ch0_0.h264")
    #vcap = cv2.VideoCapture("rtsp://192.168.62.185:1935/")
    while True:
        _, img = vcap.read()
        frame_q.append(img) #cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
except KeyboardInterrupt:
    terminate_event.set()
    logging.error("User Abort")





# video = 0

# if video:
#     fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#     video = cv2.VideoCapture('test.mp4')
#     success, image = video.read()

#     writer = cv2.VideoWriter('output.mp4', fourcc, video.get(cv2.CAP_PROP_FPS), (image.shape[0], image.shape[1]))
#     while success:
#         success,image = video.read()
#         if image is not None:
#             frame = Frame(image)
#             analysis, pose, face = sleepy_baby.process_baby_image_models(frame.w_data)
#             frame.add_analysis_frame()
#             frame.add_body_details(pose)
#             frame.add_face_details(face)
#             writer.write(frame.getAugmentedFrame())
#     cv2.destroyAllWindows()
#     writer.release()
#     video.release()
# else:
#     img = cv2.imread("test.jpg")
#     frame = Frame(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 700, 300, 1220,700)
#     analysis, pose, face = sleepy_baby.process_baby_image_models(frame.w_data)
#     frame.add_analysis_frame()
#     frame.add_body_details(pose)
#     frame.add_face_details(face)
#     cv2.imwrite("debug.jpg", frame.getAugmentedFrame())