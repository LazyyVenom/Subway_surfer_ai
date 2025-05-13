import cv2
from ultralytics import YOLO
import mss
import numpy as np
import math
import pyautogui
import random

model = YOLO('models/YoloV8/V4.pt')
names = model.names
monitor = {"top": 150, "left": 20, "width": 950, "height": 750}

sct = mss.mss()
cap = cv2.VideoCapture("videos\\final1.mp4")

while True:
    current_lane = 1
    screen_shot = sct.grab(monitor)

    frame = np.array(screen_shot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    results = model(frame, verbose=False)
    boxes = results[0].boxes

    for box, class_id in zip(boxes.xyxy, boxes.cls):
        class_id = int(class_id)
        class_name = names[class_id]

        x1, y1, x2, y2 = map(int, box)

        total_obsticales = []

        if y2 > 400 and "Train" in class_name:
            lane = math.floor(((x1+x2)/2)/310)
            print(f"Train, Lower Y (y2): {y2}")
            print(f"LANE = {lane}")
            total_obsticales.append({
                "obs": "Train",
                "lane": lane
            })


        if y2 > 400 and "Obs" in class_name:
            lane = math.floor(((x1+x2)/2)/310)
            obsi = class_name.replace('Obs_','')
            print(f"{class_name.replace('Obs_','')}, Lower Y (y2): {y2}")
            print(f"LANE = {lane}")
            total_obsticales.append({
                "obs": obsi,
                "lane": lane
            })

        class_name = class_name.replace("Obs_", "")
        class_name = "Train" if "Train" in class_name else class_name
        (text_w, text_h), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), (0, 0, 0), -1)
        cv2.putText(frame, class_name, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        #Desicion Making
        if len(total_obsticales) > 0:
            lane_info = ["","",""]
            for obstical in total_obsticales:
                if obstical["obs"] == "Train":
                    print(f"Train detected in lane {obstical['lane']}")
                    lane_info[obstical["lane"]] = "T"
                else:
                    print(f"Obstacle {obstical['obs']} detected in lane {obstical['lane']}")
                    print(obstical["lane"])
                    lane_info[obstical["lane"]] = obstical["obs"][0]

            if lane_info[current_lane] == "T":
                print("Train detected in current lane")
                if current_lane == 0:
                    print("Switching to lane 1")
                    current_lane = 1
                    pyautogui.press('right')
                elif current_lane == 1:
                    if lane_info[0] == "" and lane_info[2] == "":
                        choice = random.choice([0, 2])
                        if choice == 0:
                            pyautogui.press('left')

                        else:
                            pyautogui.press('right')
                        current_lane = choice

                    elif lane_info[2] == "T":
                        print("Switching to lane 0")
                        current_lane = 0
                        pyautogui.press('left')
                        
                    else:
                        print("Switching to lane 2")
                        current_lane = 2
                        pyautogui.press('right')
                else:
                    print("Switching to lane 0")
                    pyautogui.press('left')
                    current_lane = 1
            
            elif lane_info[current_lane] == "M" or lane_info[current_lane] == "L":
                print("Moving obstacle detected in current lane")
                pyautogui.press('up')
            
            elif lane_info[current_lane] == "H":
                print("Moving obstacle detected in current lane")
                pyautogui.press('down')


    cv2.imshow("YOLOv8 Video Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
