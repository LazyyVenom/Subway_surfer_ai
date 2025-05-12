import cv2
from ultralytics import YOLO

model = YOLO('models/YoloV8/V4.pt')

cap = cv2.VideoCapture("videos\\final1.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Video Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()