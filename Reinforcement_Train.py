import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import cv2
from ultralytics import YOLO
import mss
import math
import time
import pyautogui

# DQN Model
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return map(lambda x: torch.tensor(x, dtype=torch.float32), zip(*samples))

    def __len__(self):
        return len(self.buffer)

# Hyperparameters
STATE_DIM = 8
ACTION_DIM = 5
BATCH_SIZE = 64
GAMMA = 0.95
LR = 1e-3
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
MEMORY_SIZE = 5000
TARGET_UPDATE = 10

# Initialize
policy_net = DQN(STATE_DIM, ACTION_DIM)
target_net = DQN(STATE_DIM, ACTION_DIM)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEMORY_SIZE)

# Detection and environment setup
model = YOLO('models/YoloV8/V5.pt')
names = model.names
monitor = {"top": 200, "left": 30, "width": 850, "height": 620}
middle_lane_room = 20
sct = mss.mss()

def get_state_and_reward():
    screen_shot = sct.grab(monitor)
    frame = np.array(screen_shot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    results = model(frame, verbose=False)
    boxes = results[0].boxes

    char_details = {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "character_here": False, "avg_x": 0, "avg_y": 0}
    char_current_lane = 1  # Default to middle lane
    char_visible = 0

    obstacles_in_lanes = {
        0: {"type": 0, "dist": 999},  # Left lane
        1: {"type": 0, "dist": 999},  # Middle lane
        2: {"type": 0, "dist": 999}   # Right lane
    }
    collision_threshold = 50
    danger_threshold = 150

    for box, class_id in zip(boxes.xyxy, boxes.cls):
        class_id = int(class_id)
        class_name = names[class_id]
        x1, y1, x2, y2 = map(int, box)

        if class_name == "Char":
            char_details["x1"], char_details["y1"], char_details["x2"], char_details["y2"] = x1, y1, x2, y2
            char_details["character_here"] = True
            char_visible = 1
            char_details["avg_x"] = (x1 + x2) // 2
            char_details["avg_y"] = (y1 + y2) // 2

            lower_x_boundary = monitor["width"] // 3
            upper_x_boundary = 2 * monitor["width"] // 3
            
            if char_details["avg_x"] < lower_x_boundary:
                char_current_lane = 0
            elif char_details["avg_x"] > upper_x_boundary:
                char_current_lane = 2
            else:
                char_current_lane = 1
            break

    if not char_details["character_here"]:
        state = [char_current_lane, char_visible, 
                 obstacles_in_lanes[0]["type"], obstacles_in_lanes[0]["dist"],
                 obstacles_in_lanes[1]["type"], obstacles_in_lanes[1]["dist"],
                 obstacles_in_lanes[2]["type"], obstacles_in_lanes[2]["dist"]]
        return state, 0.0, False, frame

    for box, class_id in zip(boxes.xyxy, boxes.cls):
        class_id = int(class_id)
        class_name = names[class_id]
        x1, y1, x2, y2 = map(int, box)

        if "Train" in class_name or "Obs" in class_name:
            obs_avg_x = (x1 + x2) // 2
            obs_avg_y = (y1 + y2) // 2

            obs_lane_idx = -1
            lower_x_boundary = monitor["width"] // 3
            upper_x_boundary = 2 * monitor["width"] // 3

            if obs_avg_x < lower_x_boundary:
                obs_lane_idx = 0
            elif obs_avg_x > upper_x_boundary:
                obs_lane_idx = 2
            else:
                obs_lane_idx = 1
            
            y_distance = obs_avg_y - char_details["avg_y"]
            x_distance = obs_avg_x - char_details["avg_x"]
            distance = int(math.sqrt(x_distance**2 + y_distance**2))
            
            if y_distance < -char_details["y2"]/4:
                 continue

            obs_type_code = 0
            if "Train" in class_name:
                obs_type_code = 1
            elif class_name == "Obs_High":
                obs_type_code = 2
            elif class_name == "Obs_Low":
                obs_type_code = 3

            if obs_lane_idx != -1 and distance < obstacles_in_lanes[obs_lane_idx]["dist"]:
                obstacles_in_lanes[obs_lane_idx]["type"] = obs_type_code
                obstacles_in_lanes[obs_lane_idx]["dist"] = distance
    
    reward = 1.0
    done = False

    obstacle_in_char_lane = obstacles_in_lanes[char_current_lane]
    if obstacle_in_char_lane["type"] != 0:
        if obstacle_in_char_lane["dist"] < collision_threshold:
            reward = -100.0
            done = True
        elif obstacle_in_char_lane["dist"] < danger_threshold:
            reward = -10.0

    state = [char_current_lane, char_visible,
             obstacles_in_lanes[0]["type"], obstacles_in_lanes[0]["dist"],
             obstacles_in_lanes[1]["type"], obstacles_in_lanes[1]["dist"],
             obstacles_in_lanes[2]["type"], obstacles_in_lanes[2]["dist"]]
    
    for i in [3, 5, 7]:
        state[i] = min(state[i], 999) / 999.0

    return state, reward, done, frame

# Action selection
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, ACTION_DIM - 1)
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return policy_net(state).argmax().item()

# Training step
def train_step():
    if len(memory) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    actions = actions.long()
    dones = dones.float()

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
    max_next_q = target_net(next_states).max(1)[0]
    expected_q = rewards + (1 - dones) * GAMMA * max_next_q

    loss = nn.MSELoss()(q_values, expected_q.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def perform_action(action):
    if action == 1:
        pyautogui.press("left")
    elif action == 2:
        pyautogui.press("right")
    elif action == 3:
        pyautogui.press("up")
    elif action == 4:
        pyautogui.press("down")

def main_loop():
    global EPSILON
    episode_count = 0
    while True:
        episode_count += 1
        print(f"Starting Episode: {episode_count}")

        current_total_reward = 0
        steps_in_episode = 0

        while True:
            state, reward, done, frame = get_state_and_reward()
            action = select_action(state, EPSILON)
            perform_action(action)
            
            time.sleep(0.05)

            next_state, next_reward, next_done, next_frame = get_state_and_reward()
            
            current_total_reward += reward
            steps_in_episode += 1

            memory.push(state, action, reward, next_state, float(done or next_done))
            
            train_step()

            if steps_in_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
            
            debug_text = f"Action: {action}, Reward: {reward:.2f}, Epsilon: {EPSILON:.3f}"
            cv2.putText(next_frame, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            state_text = f"State: {[f'{s:.2f}' if isinstance(s, float) else s for s in state]}"
            cv2.putText(next_frame, state_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)

            cv2.imshow("YOLOv8 RL Detection", next_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or done or next_done:
                print(f"Episode {episode_count} finished after {steps_in_episode} steps. Total Reward: {current_total_reward:.2f}, Epsilon: {EPSILON:.3f}")
                if done or next_done:
                    pyautogui.press("space")
                    time.sleep(1)
                break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()