import gymnasium as gym


from tabular.TabularAgent import BlackjackAgent
from tabular.TabularTrain import train
from helpers.plot import plot_rewards, create_q_table_subplots
import numpy as np



# We define some hyperparameters that we will use during the training
lr = 0.00001
epsilon_decay = 1e-5
initial_epsilon = 1


# Although after that epsilon becomes 0 we stop to go through the exploring
# phase, we need way more episodes probably because we have a lot of the same
# values and maxarg needs to choose from there


#n_episodes = round(initial_epsilon / epsilon_decay)
n_episodes = 300000

env = gym.make('Blackjack-v1', sab=True)
agent = BlackjackAgent(env, lr=lr, initial_epsilon = initial_epsilon, epsilon_decay=epsilon_decay)

rewards = train(agent, env, n_episodes)

print(f"Epsilon: {initial_epsilon}")
print(f"Epsilon decay: {epsilon_decay}")
print(f"Number of episodes: {n_episodes}")
print(f"Average reward per episode: {np.mean(rewards)}")

plot_rewards("tabular_rewards_0.00001", rewards)
create_q_table_subplots("tabular_0.00001",agent)

