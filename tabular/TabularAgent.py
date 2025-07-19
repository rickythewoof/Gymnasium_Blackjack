from collections import defaultdict
import numpy as np

class BlackjackAgent():

    def __init__(self, env, lr=1e-3, initial_epsilon=1, epsilon_decay=1e-3):
        """
        Initialize an Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.
        """

        # We don't need to have q-table over the observation, as it
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n)) # maps a state to action values
        self.lr = lr
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay

    def get_action(self, state: tuple[int, int, int]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        and a random action with probability epsilon to ensure exploration.
        """
        # Explore = with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()

        # Exploit = with probability (1 - epsilon) act greedily
        else:
            action = np.argmax(self.q_values[state])
        return action

    def update(self, state: tuple[int, int, int], action: tuple[int],
               reward, next_state: tuple[int, int, int], done):
        """
        Updates the Q-value of an action.
        Q(s,a) = (1-lr)*Q(s,a) + lr[reward + (1-done)*max(Q(s',a')]
        """
        old_q_value = self.q_values[state][action]
        max_future_q = np.max(self.q_values[next_state])
        target = reward + self.lr * max_future_q * (1 - done)
        self.q_values[state][action] = (1 - self.lr) * old_q_value + self.lr * target

    def decay_epsilon(self):
        self.epsilon = max(0, self.epsilon - self.epsilon_decay)