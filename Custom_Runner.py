import cv2
from ultralytics import YOLO
import mss
import numpy as np
import math
import time
import pyautogui

model = YOLO('models/YoloV8/V5.pt')
names = model.names
monitor = {"top": 200, "left": 30, "width": 850, "height": 620}
middle_lane_room = 20

sct = mss.mss()
cap = cv2.VideoCapture("videos\\final1.mp4")
current_lane = 1
character_last_seen = time.time()

while True:
    screen_shot = sct.grab(monitor)
    frame = np.array(screen_shot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    total_obstacles = []

    char_details = {
        "x1": 0,
        "y1": 0,
        "x2": 0,
        "y2": 0,
        "character_here": False
    }
    for box, class_id in zip(boxes.xyxy, boxes.cls):
        class_id = int(class_id)
        class_name = names[class_id]
        x1, y1, x2, y2 = map(int, box)

        if class_name == "Char":
            character_last_seen = time.time()

            char_details["x1"] = x1
            char_details["y1"] = y1
            char_details["x2"] = x2
            char_details["y2"] = y2
            char_details["character_here"] = True

            char_x_avg = (x1 + x2) // 2
            lower_x = monitor["width"] // 2 - middle_lane_room
            upper_x = monitor["width"] // 2 + middle_lane_room
            
            if char_x_avg < lower_x:
                lane = 0
            elif char_x_avg > upper_x:
                lane = 2
            else:
                lane = 1

            current_lane = lane

    for box, class_id in zip(boxes.xyxy, boxes.cls):
        class_id = int(class_id)
        class_name = names[class_id]
        x1, y1, x2, y2 = map(int, box)

        if "Train" in class_name or "Obs" in class_name:
            x1_avg = (x1 + x2) // 2
            y1_avg = (y1 + y2) // 2

            char_x_avg = (char_details["x1"] + char_details["x2"]) // 2
            obs_x_avg = (x1 + x2) // 2
            
            y_distance = y1_avg - char_details["y1"]
            x_distance = obs_x_avg - char_x_avg
            distance = math.sqrt(x_distance**2 + y_distance**2)
            
            measured_angle = math.degrees(math.atan2(y_distance, x_distance))
            if -80 > measured_angle > -100 and distance < 170:
                print(f"Class: {class_name}, Angle: {measured_angle}, Distance: {distance}")

                if "Train" in class_name:
                    if current_lane == 0:
                        pyautogui.press("right")
                    elif current_lane == 2:
                        pyautogui.press("left")
                    else:
                        pyautogui.press("left")
                
                else:
                    if class_name == "Obs_High":
                        pyautogui.press("down")
                    elif class_name == "Obs_Low":
                        pyautogui.press("up")
                    else:
                        pyautogui.press("up")

            # print(f"Angle: {measured_angle}, Distance: {math.sqrt(x_distance**2 + y_distance**2)}")

    cv2.imshow("YOLOv8 Video Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
