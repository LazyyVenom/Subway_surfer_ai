import cv2
import numpy as np
from ultralytics import YOLO
import mss

model = YOLO('models\YoloV8\V3.pt')

monitor = {"top": 0, "left": 0, "width": 1280, "height": 720}

sct = mss.mss()

while True:
    screen_shot = sct.grab(monitor)
    
    frame = np.array(screen_shot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Drop alpha channel

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Screen Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()