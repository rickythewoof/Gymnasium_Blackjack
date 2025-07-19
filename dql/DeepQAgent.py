import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import random
from collections import deque
from tqdm import tqdm




class DeepQNetwork(nn.Module):
  def __init__(self, input_size, output_size):
    super(DeepQNetwork, self).__init__()

    self.fc1 = nn.Linear(input_size, 64)
    self.fc2 = nn.Linear(64, 128)
    self.out = nn.Linear(128, output_size)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.out(x)

class ReplayBuffer():
  def __init__(self, cap=10000):
    self.buffer = deque(maxlen=cap)

  def push(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def sample(self, batch_size):
    state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
    return state, action, reward, next_state, done

  def __len__(self):
    return len(self.buffer)

def train(agent, env, n_episodes):
    episode_rewards_dqn = []
    for episode in tqdm(range(n_episodes)):
        state, info = env.reset()
        done = False
        ep_rew = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store the experience in the replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Update the Q-network
            agent.update()

            ep_rew += reward
            state = next_state

        episode_rewards_dqn.append(ep_rew)

        # Decay epsilon
        agent.decay_epsilon()

        # Update the target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
    return episode_rewards_dqn

class BlackjackDQN:
    def __init__(self, env, device, lr, epsilon, epsilon_decay, batch_size, replay_buffer_size):
        self.env = env
        self.device = device
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Calculate input size based on the Tuple observation space
        # The observation space is a Tuple(Discrete(32), Discrete(11), Discrete(2))
        # We need to flatten this to 3 for the neural network input
        input_size = 3

        self.q_network = DeepQNetwork(input_size, env.action_space.n).to(self.device)
        self.target_network = DeepQNetwork(input_size, env.action_space.n).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            # Convert tuple state to a flattened numpy array for the network
            state_array = np.array(state, dtype=np.float32)
            state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert tuple states to flattened numpy arrays for the network
        states = torch.tensor(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states, dtype=np.float32)).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)


        # Compute Q-values for current states
        q_values = self.q_network(states).gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.lr * next_q_values * (1 - dones)

        # Compute loss and update the network
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(0, self.epsilon - self.epsilon_decay)