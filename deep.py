import gymnasium as gym
from dql.DeepQAgent import BlackjackDQN, train
import torch
from helpers.plot import plot_rewards, create_dqn_policy_subplots


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Using device", device)

num_episodes = 100000

epsilon_start = 1.0
epsilon_decay = 1e-4
lr = 0.01
batch_size = 256
replay_buffer_size = num_episodes
env = gym.make("Blackjack-v1")
agent = BlackjackDQN(env, device, lr, epsilon_start, epsilon_decay, batch_size, replay_buffer_size)


rewards = train(agent, env, num_episodes)

create_dqn_policy_subplots("dql-table_0.01", device, agent)
plot_rewards("dql/dql_rewards_0.01", rewards)