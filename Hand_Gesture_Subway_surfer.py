import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
# Lower the detection confidence for better sensitivity
hands = mp_hands.Hands(static_image_mode=False, 
                       max_num_hands=1, 
                       min_detection_confidence=0.5,  # Lower threshold for easier detection
                       min_tracking_confidence=0.5)   # Lower tracking threshold too
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# For smoother control
last_action_time = time.time()
cooldown = 0.5

# Gesture thresholds
thresholds = {
    'up': 10,     # Hand position above this y-value triggers jump
    'down': 10,   # Hand position below this y-value triggers slide
    'left': -0.1,  # Negative x-tilt beyond this value triggers left
    'right': 0.1   # Positive x-tilt beyond this value triggers right
}

# Calibration variables
calibration_mode = False
calibration_samples = []
calibration_count = 0
max_calibration_samples = 30

# Flag to track keys currently pressed
keys_pressed = set()

def detect_gesture(hand_landmarks, img_height, img_width):
    """Detects gesture based on hand landmarks position and the plus-sign grid"""
    # Use multiple landmarks for more robust detection
    # Calculate the center of the hand using multiple points
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Average position from multiple landmarks for stability
    x_pos = (wrist.x + index_mcp.x + middle_mcp.x + ring_mcp.x) / 4
    y_pos = (wrist.y + index_mcp.y + middle_mcp.y + ring_mcp.y) / 4
    
    # Convert normalized coordinates to pixel values
    px = int(x_pos * img_width)
    py = int(y_pos * img_height)
    
    # Center coordinates (where the plus sign intersects)
    center_x = img_width // 2
    center_y = img_height // 2
    
    # Determine position relative to the plus sign
    if py < center_y - 20:  # Some buffer to prevent flickering
        return "up"
    elif py > center_y + 20:
        return "down"
    elif px < center_x - 20:
        return "left"
    elif px > center_x + 20:
        return "right"
    else:
        return "center"  # Hand is in the center region

def draw_plus_grid(img):
    """Draw a plus sign on the image to guide gesture detection"""
    height, width, _ = img.shape
    center_x = width // 2
    center_y = height // 2
    
    # Draw horizontal line
    cv2.line(img, (0, center_y), (width, center_y), (255, 255, 255), 2)
    
    # Draw vertical line
    cv2.line(img, (center_x, 0), (center_x, height), (255, 255, 255), 2)
    
    # Draw center point
    cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
    
    # Draw labels
    cv2.putText(img, "UP", (center_x - 20, center_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "DOWN", (center_x - 30, center_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "LEFT", (center_x - 90, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "RIGHT", (center_x + 30, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def highlight_active_region(img, gesture):
    """Highlight the active region based on the current gesture"""
    height, width, _ = img.shape
    center_x = width // 2
    center_y = height // 2
    
    # Semi-transparent overlay
    overlay = img.copy()
    
    if gesture == "up":
        cv2.rectangle(overlay, (0, 0), (width, center_y), (0, 255, 0), -1)
    elif gesture == "down":
        cv2.rectangle(overlay, (0, center_y), (width, height), (0, 255, 0), -1)
    elif gesture == "left":
        cv2.rectangle(overlay, (0, 0), (center_x, height), (0, 255, 0), -1)
    elif gesture == "right":
        cv2.rectangle(overlay, (center_x, 0), (width, height), (0, 255, 0), -1)
    elif gesture == "center":
        # Small rectangle in the center
        size = 40
        cv2.rectangle(overlay, 
                     (center_x - size, center_y - size), 
                     (center_x + size, center_y + size), 
                     (0, 255, 0), -1)
    
    # Apply the overlay with transparency
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def calibrate(hand_landmarks):
    """Collect calibration data"""
    global calibration_samples, calibration_count
    
    # Extract key landmarks
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    
    # Calculate hand center and tilt
    hand_y = (wrist.y + index_tip.y + middle_tip.y) / 3
    tilt_x = index_tip.x - wrist.x
    
    # Add to calibration samples
    calibration_samples.append((hand_y, tilt_x))
    calibration_count += 1
    
    # If we have enough samples, calculate thresholds
    if calibration_count >= max_calibration_samples:
        calculate_thresholds()
        return True
    return False

def calculate_thresholds():
    """Calculate gesture thresholds from calibration data"""
    global thresholds, calibration_samples, calibration_mode
    
    # Extract all hand_y and tilt_x values
    hand_y_values = [sample[0] for sample in calibration_samples]
    tilt_x_values = [sample[1] for sample in calibration_samples]
    
    # Calculate statistics
    mean_y = np.mean(hand_y_values)
    std_y = np.std(hand_y_values)
    mean_x = np.mean(tilt_x_values)
    std_x = np.std(tilt_x_values)
    
    thresholds['up'] = mean_y - std_y
    thresholds['down'] = mean_y + std_y
    thresholds['left'] = mean_x - std_x
    thresholds['right'] = mean_x + std_x
    
    print(f"Calibration complete. Thresholds: {thresholds}")
    calibration_mode = False

def show_detection_status(img, is_hand_detected):
    """Display whether a hand is currently being detected"""
    status_color = (0, 255, 0) if is_hand_detected else (0, 0, 255)  # Green if detected, Red if not
    status_text = "Hand Detected" if is_hand_detected else "No Hand Detected"
    cv2.putText(img, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

try:
    print("Hand Gesture Control for Subway Surfers")
    print("Press 'c' to calibrate, 'q' to quit")
    print("Make sure your hand is well-lit and clearly visible to the camera")
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break
        
        img = cv2.flip(img, 1)
        
        # Get image dimensions
        img_height, img_width, _ = img.shape
        
        # Draw the plus grid for visual guidance
        draw_plus_grid(img)
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb_img)
        
        action = "none"
        hand_detected = False
        
        status_text = "Press 'c' to calibrate" if not calibration_mode else f"Calibrating: {calibration_count}/{max_calibration_samples}"
        
        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Make landmarks more visible
                draw_params = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4)
                connection_params = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                         draw_params, connection_params)
                
                if calibration_mode:
                    calibration_complete = calibrate(hand_landmarks)
                    if calibration_complete:
                        status_text = "Calibration complete"
                else:
                    # Pass image dimensions to the detect_gesture function
                    action = detect_gesture(hand_landmarks, img_height, img_width)
                    
                    # Highlight the active region
                    highlight_active_region(img, action)
                    
                    status_text = f"Gesture: {action}"
                    
                    # Draw the position of detected hand center
                    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    px = int(index_mcp.x * img_width)
                    py = int(index_mcp.y * img_height)
                    cv2.circle(img, (px, py), 10, (255, 0, 255), -1)  # Magenta dot for hand center
        
        cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show whether a hand is detected
        show_detection_status(img, hand_detected)
        
        cv2.imshow("Hand Gesture Control", img)
        
        # Only trigger keys if a hand is detected
        current_time = time.time()
        if hand_detected and not calibration_mode and current_time - last_action_time > cooldown:
            for key in keys_pressed:
                pyautogui.keyUp(key)
            keys_pressed.clear()
            
            if action == "left":
                pyautogui.keyDown('left')
                keys_pressed.add('left')
                last_action_time = current_time
            elif action == "right":
                pyautogui.keyDown('right')
                keys_pressed.add('right')
                last_action_time = current_time
            elif action == "up":
                pyautogui.keyDown('up')
                keys_pressed.add('up')
                last_action_time = current_time
            elif action == "down":
                pyautogui.keyDown('down')
                keys_pressed.add('down')
                last_action_time = current_time
        
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            calibration_mode = True
            calibration_samples = []
            calibration_count = 0
            print("Starting calibration. Please move your hand naturally in the center position.")

except Exception as e:
    print(f"Error: {e}")

finally:
    for key in keys_pressed:
        pyautogui.keyUp(key)
    
    cap.release()
    cv2.destroyAllWindows()