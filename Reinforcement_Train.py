import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

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
STATE_DIM = 5
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

def game_loop(observation, reward, done):
    global EPSILON

    state = np.array(observation, dtype=np.float32)
    action = select_action(state, EPSILON)

    perform_action(action)

    new_observation = get_new_state()
    memory.push(state, action, reward, new_observation, float(done))

    train_step()

    if len(memory) % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

def perform_action(action):
    # 0: Stay, 1: Left, 2: Right, 3: Jump, 4: Duck
    print("Action:", action)

def get_new_state():
    return [1, 0, 1, 0, 2]