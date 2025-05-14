import cv2
from ultralytics import YOLO
import mss
import numpy as np
import math
import time
import pyautogui

# Configuration parameters
model = YOLO('models/YoloV8/V5.pt')
names = model.names
monitor = {"top": 200, "left": 30, "width": 850, "height": 620}
middle_lane_room = 20

# Keep only relevant configuration
CHARACTER_LOST_TIMEOUT = 5.0

# Performance tracking
fps_start_time = time.time()
fps_counter = 0
fps = 0

sct = mss.mss()
cap = cv2.VideoCapture("videos\\final1.mp4")
current_lane = 1
character_last_seen = time.time()

char_details = {
    "x1": 0, "y1": 0, "x2": 0, "y2": 0,
    "character_here": False
}

while True:
    # Capture screen
    screen_shot = sct.grab(monitor)
    frame = np.array(screen_shot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    display_frame = frame.copy()
    
    char_details["character_here"] = False
    
    # Apply YOLO detection
    results = model(frame, verbose=False)
    
    # Update FPS counter
    fps_counter += 1
    if (time.time() - fps_start_time) > 1.0:
        fps = fps_counter / (time.time() - fps_start_time)
        fps_counter = 0
        fps_start_time = time.time()
    
    display_frame = results[0].plot()
    boxes = results[0].boxes
    
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Character detection logic
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

    # Obstacle detection and avoidance logic
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
            if -80 > measured_angle > -100 and distance < 180:
                print(f"Class: {class_name}, Angle: {measured_angle}, Distance: {distance}")
                
                # Draw a line between character and obstacle
                char_center = (char_x_avg, char_details["y1"])
                obstacle_center = (obs_x_avg, y1_avg)
                
                # Use different colors based on object type
                if "Train" in class_name:
                    line_color = (0, 0, 255)  # Red for trains
                elif class_name == "Obs_High":
                    line_color = (255, 0, 0)  # Blue for high obstacles
                elif class_name == "Obs_Low":
                    line_color = (0, 255, 255)  # Yellow for low obstacles
                else:
                    line_color = (255, 0, 255)  # Magenta for other obstacles
                
                # Draw the line with thickness proportional to proximity (closer = thicker)
                line_thickness = max(1, min(5, int(180 / distance * 2)))
                cv2.line(display_frame, char_center, obstacle_center, line_color, line_thickness)
                
                # Display distance and angle information next to the line
                text_pos = ((char_center[0] + obstacle_center[0]) // 2, 
                           (char_center[1] + obstacle_center[1]) // 2 - 10)
                cv2.putText(display_frame, f"{distance:.1f}", text_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1)

                if "Train" in class_name:
                    if current_lane == 0:
                        pyautogui.press("right")
                        print("Pressed right")
                    elif current_lane == 2:
                        pyautogui.press("left")
                        print("Pressed left")
                    else:
                        random_choice = np.random.choice(["left", "right"])
                        print(f"Choice: {random_choice}")
                        if random_choice == "left":
                            pyautogui.press("left")
                        else:
                            pyautogui.press("right")
                
                elif distance < 160:
                    if class_name == "Obs_High":
                        pyautogui.press("down")
                        print("Pressed down")
                    elif class_name == "Obs_Low":
                        pyautogui.press("up")
                        print("Pressed up")
                    else:
                        random_choice = np.random.choice(["down", "up"])
                        print(f"Choice: {random_choice}")
                        if random_choice == "down":
                            pyautogui.press("down")
                        else:
                            pyautogui.press("up")

    # Check if character is lost
    if not char_details["character_here"] and time.time() - character_last_seen > CHARACTER_LOST_TIMEOUT:
        text = "I Lost Sorry :("
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        color = (0, 0, 255)
        thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (display_frame.shape[1] - text_size[0]) // 2
        text_y = (display_frame.shape[0] + text_size[1]) // 2
        cv2.putText(display_frame, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    cv2.imshow("YOLOv8 Video Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
