import cv2
from ultralytics import YOLO
import mss
import numpy as np

model = YOLO('models/YoloV8/V4.pt')

names = model.names

monitor = {"top": 150, "left": 20, "width": 950, "height": 750}

sct = mss.mss()

cap = cv2.VideoCapture("videos\\final1.mp4")

while True:
    screen_shot = sct.grab(monitor)

    frame = np.array(screen_shot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    results = model(frame, verbose=False)
    boxes = results[0].boxes

    for box, class_id in zip(boxes.xyxy, boxes.cls):
        class_id = int(class_id)
        class_name = names[class_id]

        x1, y1, x2, y2 = map(int, box)

        if y2 > 400 and "Train" in class_name:
            print(f"Train, Lower Y (y2): {y2}")


        if y2 > 380 and "Obs" in class_name:
            print(f"{class_name.replace("Obs_","")}, Lower Y (y2): {y2}")

        (text_w, text_h), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), (0, 0, 0), -1)
        cv2.putText(frame, class_name, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("YOLOv8 Video Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
