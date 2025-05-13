import cv2
from ultralytics import YOLO

model = YOLO('models/YoloV8/V5.pt')

# cap = cv2.VideoCapture("videos\\final1.mp4")
cap = cv2.VideoCapture(r"C:\Users\Anubhav Choubey\Videos\Recording 2025-05-13 210352.mp4")

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