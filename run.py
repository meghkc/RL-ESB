# run.py
import numpy as np
import matplotlib.pyplot as plt
from environment import BusSchedulingEnv
from ppo_agent import PPOAgent
from config import STATE_DIM, ACTION_DIM, NUM_EPISODES

def train():
    env = BusSchedulingEnv()
    env.print_problem()
    
    agent = PPOAgent(STATE_DIM, ACTION_DIM, hidden_size=64)
    episode_rewards = []
    avg_rewards = []  # average reward for each block of 50 episodes
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        trajectory = []  # list of (state, action, log_prob, reward, done)
        total_reward = 0
        
        while not done:
            action, log_prob, _ = agent.policy.act(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, log_prob, reward, done))
            state = next_state
            total_reward += reward
            
        episode_rewards.append(total_reward)
        agent.update(trajectory)
        
        # Every 50 episodes, calculate the average reward over the last 50 episodes.
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_rewards.append(avg_reward)
            print(f"Episode {episode+1}/{NUM_EPISODES} - Average Reward: {avg_reward:.2f}")
    
    return episode_rewards, avg_rewards, agent, env

def evaluate(agent, env):
    state = env.reset()
    done = False
    while not done:
        action, _, _ = agent.policy.act(state)
        state, _, done, _ = env.step(action)
    env.print_solution()

if __name__ == '__main__':
    rewards, avg_rewards, agent, env = train()
    
    episodes = np.arange(1, NUM_EPISODES+1)
    
    # Plot the reward for each episode
    plt.plot(episodes, rewards, label="Episode Reward", alpha=0.5)
    
    # Compute x-values for average rewards: these occur every 50 episodes.
    avg_episode_nums = np.arange(50, NUM_EPISODES+1, 50)
    plt.plot(avg_episode_nums, avg_rewards, label="Average Reward (per 50 eps)", color="red", linewidth=2)
    
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Progress over Training Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    evaluate(agent, env)
