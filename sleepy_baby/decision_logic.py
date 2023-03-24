import logging
from collections import deque
import statistics

import cv2
import numpy as np


class DecisionLogic:
    
    # TODO: break up this class, so big ew

    # General high level heuristics:
    # 1) no eyes -> no body found -> baby is awake
    # 2) no eyes -> body found -> moving -> baby is awake
    # 3) no eyes -> body found -> not moving -> baby is sleeping
    # 4) eyes -> eyes open -> baby is awake (disregard body movement)
    # 5) eyes -> eyes closed -> movement -> baby is awake
    # 6) eyes -> eyes closed -> no movement -> baby is asleep
    # 7) eyes -> eyes closed -> mouth open -> baby is awake

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__qualname__)
        self.eyes_open_q = deque(maxlen=30)
        self.awake_q = deque(maxlen=40)
        self.movement_q = deque(maxlen=40)
        self.eyes_open_state = False
        self.is_awake = False
        self.body_found = False
        self.eyes_found = False
        self.avg_awake = 0


    def push(self, analysis):
        self.body_found = analysis['body_detected']
        self.eyes_found = analysis['face_detected']
        if self.body_found:
            self.movement_q.append((analysis['left_wrist_coords'], analysis['right_wrist_coords']))
            if self.eyes_found:
                if analysis['eyes_open'] is False:
                    self.eyes_open_q.append(1 if analysis['mouth_open'] else 0)
                else:
                    self.eyes_open_q.append(1)
            #no_eyes_found
        #no_body_found
        

    def update(self, eyes_threshold:float=0.75, wrist_threshold:int=25) -> None: #every second
        
        if self.body_found is False: #throttled_handle_no_body_found
            self.awake_q.append(1)
        if (self.eyes_found is False) and (len(self.eyes_open_q)>0): #throttled_handle_no_eyes_found
            self.eyes_open_q.popleft()

        #self.awake_voting_logic()
        if len(self.eyes_open_q) > self.eyes_open_q.maxlen/2:
            avg = sum(self.eyes_open_q) / len(self.eyes_open_q)
            if avg > 0.75: #eyes_open
                self.eyes_open_state = True
                self.logger.info("Eyes Open: vote awake")
                self.awake_q.append(1)
            else:
                self.eyes_open_state = False
                self.awake_q.append(0)
                self.logger.info("Eyes closed: vote sleeping")
        else:
            self.logger.debug("Not voting on eyes, eye que is too short.")
        
        #self.movement_voting_logic(body_found)
        if self.body_found is False:
            self.logger.debug("No body found, depreciate movement queue.")
            if len(self.movement_q)>0:
                self.movement_q.popleft()
        elif (movement_list_len := len(self.movement_q)) > 5:
            positions = np.reshape(self.movement_q, (movement_list_len, 4)).T
            st_dev = [statistics.pstdev(pos) for pos in positions]
            avg_std = sum(st_dev)/4
            
            if int(avg_std) < wrist_threshold:
                self.logger.info('No movement, vote sleeping')
                self.awake_q.append(0)
            else:
                print("Movement, vote awake")
                self.logger.info("Movement, vote awake")
                self.awake_q.append(1)

        #self.set_wakeness_status()
        if len(self.awake_q)>0:
            self.avg_awake = sum(self.awake_q) / len(self.awake_q)
            if self.avg_awake >= 0.6 and self.is_awake == False:
                self.logger.info("Awake Event")
                self.is_awake = True
                #self.need_to_clean_this_up(True) #TODO
            elif self.avg_awake <0.6 and self.is_awake == True:
                self.logger.info("Sleep Event")
                self.is_awake = False
                #self.need_to_clean_this_up(True) #TODO

        #self.periodic_wakeness_check()
        
       
    # This is placeholder until improve sensitivity of transitioning between waking and sleeping.
    # Explanation: Sometimes when baby is waking up, he'll open and close his eyes for a couple of minutes...
    # TODO: Fine-tune sensitivity of voting, for now, don't allow toggling between wake & sleep within N seconds
    #@debounce(180)
    def need_to_clean_this_up(self, wake_status, img):
        str_timestamp = str(int(time.time()))
        sleep_data_base_path = os.getenv("SLEEP_DATA_PATH")
        p = sleep_data_base_path + '/' + str_timestamp + '.png'
        if wake_status: # woke up
            log_string = "1," + str_timestamp + "\n"
            print(log_string)
            logging.info(log_string)
            with open(sleep_data_base_path + '/sleep_logs.csv', 'a+', encoding="utf-8") as f:
                f.write(log_string)
            cv2.imwrite(p, img) # store off image of when wake/sleep event occurred. Can help with debugging issues

            # if daytime, send phone notification if baby woke up
            # now = datetime.datetime.now()
            # now_time = now.time()
            # if now_time >= ti(7,00) or now_time <= ti(22,00): # day time
            #     Thread(target=telegram_send.send(messages=["Baby woke up."]), daemon=True).start()

            self.is_awake = True

            if os.getenv("OWL", 'False').lower() in ('true', '1'):
                print("MOVE & MAKE NOISE")
                logging.info("MOVE & MAKE NOISE")
                time.sleep(5)
                self.ser.write(bytes(str(999999) + "\n", "utf-8"))
                self.cast_service.play_sound()
        else: # fell asleep
            log_string = "0," + str_timestamp + "\n"
            print(log_string)
            logging.info(log_string)
            with open(sleep_data_base_path + '/sleep_logs.csv', 'a+', encoding="utf-8") as f:
                f.write(log_string)
            cv2.imwrite(p, img)
            self.is_awake = False

        # now = datetime.datetime.now()
        # now_time = now.time()
        # if now_time >= ti(22,00) or now_time <= ti(8,00): # night time
        #     set_hatch(self.is_awake)


   


    def frame_logic(self, raw_img):
        img = raw_img

        debug_img = img.copy()
        img.flags.writeable = False
        converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # beef
        res = self.process_baby_image_models(converted_img, debug_img)
        debug_img = res[0]
        body_found = res[1]

        self.awake_voting_logic(debug_img)
        self.movement_voting_logic(debug_img, body_found)
        self.set_wakeness_status(debug_img)
        self.periodic_wakeness_check()

        if os.getenv("DEBUG", 'False').lower() in ('true', '1'):
            avg_awake = sum(self.awake_q) / len(self.awake_q)
            
            # draw progress bar
            bar_y_offset = 0
            bar_y_offset = 100

            bar_width = 200
            w = img.shape[1]
            start_point = (int(w/2 - bar_width/2), 350 + bar_y_offset)

            end_point = (int(w/2 + bar_width/2), 370 + bar_y_offset)
            adj_avg_awake = 1.0 if avg_awake / .6 >= 1.0 else avg_awake / .6
            progress_end_point = (int(w/2 - bar_width/2 + (bar_width*(adj_avg_awake))), 370 + bar_y_offset)

            color = (255, 255, 117)
            progress_color = (0, 0, 255)
            thickness = -1

            debug_img = cv2.rectangle(debug_img, start_point, end_point, color, thickness)
            debug_img = cv2.rectangle(debug_img, start_point, progress_end_point, progress_color, thickness)
            display_perc = int((avg_awake * 100) / 0.6)
            display_perc = 100 if display_perc >= 100 else display_perc
            debug_img = cv2.putText(debug_img, str(display_perc) + "%", (int(w/2 - bar_width/2), 330 + bar_y_offset), 2, 1, (255,0,0), 2, 2)
            debug_img = cv2.putText(debug_img, "Awake", (int(w/2 - bar_width/2 + 85), 330 + bar_y_offset), 2, 1, (255,0,0), 2, 2)

        return debug_img