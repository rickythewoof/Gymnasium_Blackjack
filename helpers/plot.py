import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
import torch

def create_q_table_subplots(filename, agent):
    """
    Plots the policy (best action) based on the Q-table for different states
    in two subplots, one for playable ace and one without.
    """
    player_sum_range = range(4, 22)
    dealer_card_range = range(1, 11)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, playable_ace in enumerate([False, True]):
        # Create a grid for plotting
        grid = np.zeros((len(player_sum_range), len(dealer_card_range)))

        for r, player_sum in enumerate(player_sum_range):
            for c, dealer_card in enumerate(dealer_card_range):
                state = (player_sum, dealer_card, playable_ace)
                # Get the index of the best action (0: stick, 1: hit)
                best_action = np.argmax(agent.q_values[state])
                grid[r, c] = best_action

        sns.heatmap(
            grid,
            ax=axes[i],
            annot=True,
            cmap="coolwarm",
            xticklabels=dealer_card_range,
            yticklabels=player_sum_range,
            cbar=False # Remove heatmap legend
        )
        axes[i].set_title(f"Optimal Policy - Playable Ace: {playable_ace}")
        axes[i].set_xlabel("Dealer's Showing Card")
        axes[i].set_ylabel("Player's Sum")
        axes[i].tick_params(axis='y', rotation=0)
        axes[i].invert_yaxis()


    # Add a single legend for both subplots
    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Stick'),
        Patch(facecolor='red', edgecolor='black', label='Hit')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to make space for the legend

    plt.savefig(f"./out/tabular/{filename}.png")



def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

def plot_rewards(filename, rewards):
    '''
        Plots the rewards over time, smoothed over a window size of 1000 episodes.
    '''
    plt.figure(figsize=(10, 6))

    plt.plot(
        moving_average(rewards, window_size=1000), color="blue", label="Smoothed Rewards"
    )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Average Performance Over Time")
    plt.legend()
    plt.savefig(f"./out/{filename}.png")


def create_dqn_policy_subplots(filename, device, agent):
    """
    Plots the optimal policy (best action) based on the trained DQN agent
    for different Blackjack states in two subplots.
    """
    player_sum_range = range(4, 22)
    dealer_card_range = range(1, 11)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, playable_ace in enumerate([False, True]):
        # Create a grid for plotting the policy
        grid = np.zeros((len(player_sum_range), len(dealer_card_range)))

        for r, player_sum in enumerate(player_sum_range):
            for c, dealer_card in enumerate(dealer_card_range):
                state = (player_sum, dealer_card, playable_ace)
                # Convert the state tuple to a flattened numpy array and then to a tensor
                state_array = np.array(state, dtype=np.float32)
                state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(device)

                # Get the Q-values for this state using the trained q_network
                with torch.no_grad():
                    q_values = agent.q_network(state_tensor)

                # Get the index of the best action (0: stick, 1: hit)
                best_action = torch.argmax(q_values).item()
                grid[r, c] = best_action

        sns.heatmap(
            grid,
            ax=axes[i],
            annot=True,
            cmap="coolwarm",
            xticklabels=dealer_card_range,
            yticklabels=player_sum_range,
            cbar=False # Remove heatmap legend
        )
        axes[i].set_title(f"Optimal Policy from DQN - Playable Ace: {playable_ace}")
        axes[i].set_xlabel("Dealer's Showing Card")
        axes[i].set_ylabel("Player's Sum")
        axes[i].tick_params(axis='y', rotation=0)
        axes[i].invert_yaxis()

    # Add a single legend for both subplots
    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Stick'),
        Patch(facecolor='red', edgecolor='black', label='Hit')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to make space for the legend
    plt.savefig(f"./out/dql/{filename}.png")
