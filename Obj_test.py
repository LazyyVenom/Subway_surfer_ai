import cv2
import numpy as np
from ultralytics import YOLO
import mss

# Load your trained YOLOv8 model
model = YOLO('path/to/your/best.pt')  # Replace with your weights

# Define the screen capture area (you can customize this)
monitor = {"top": 0, "left": 0, "width": 1280, "height": 720}

# Initialize mss for screen capturing
sct = mss.mss()

while True:
    # Capture screen
    screen_shot = sct.grab(monitor)
    
    # Convert to a NumPy array (RGB image)
    frame = np.array(screen_shot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Drop alpha channel

    # Run YOLOv8 inference
    results = model(frame, verbose=False)

    # Visualize detections
    annotated_frame = results[0].plot()

    # Display the result
    cv2.imshow("YOLOv8 Screen Inference", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
