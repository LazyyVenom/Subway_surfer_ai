import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('models/YoloV8/V3.pt')

# Open webcam (use 0 for default webcam, or replace with video path)
cap = cv2.VideoCapture("videos\\final1.mp4")  # or 'video.mp4' for file input

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if video ends or camera disconnects

    # Inference with YOLOv8
    results = model(frame, verbose=False)

    # Plot the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame~~
    cv2.imshow("YOLOv8 Video Detection", annotated_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and close windows
cap.release()
cv2.destroyAllWindows()
