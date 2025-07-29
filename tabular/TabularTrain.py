from tqdm import tqdm


def train(agent, env, n_episodes):
    episode_rewards = []
    for episode in tqdm(range(n_episodes)):
        state, info = env.reset()
        done = False
        ep_rew = 0
        # play one episode
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated # set done=True if episode ended early
            agent.update(state, action, reward, next_state, done)

            ep_rew += reward

            state = next_state
        agent.decay_epsilon()
        episode_rewards.append(ep_rew)
    return episode_rewards

