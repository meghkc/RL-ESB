"""
Main script for training and evaluating the PPO agent.
"""

import argparse
import time

import numpy as np

from config import (
    DATASET_INSTANCE,
    DATASET_ROOT,
    DATASET_SPLIT,
    DATASET_SUBSET,
    DATA_SOURCE,
    HIDDEN_SIZE,
    NUM_EPISODES,
)
from environment import BusSchedulingEnv
from ppo_agent import PPOAgent
from utils import create_results_directory, plot_rewards, write_results


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO for electric bus scheduling")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Number of training episodes")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument(
        "--data-source",
        type=str,
        default=DATA_SOURCE,
        choices=["synthetic", "juliette"],
        help="Environment data source",
    )
    parser.add_argument("--dataset-root", type=str, default=DATASET_ROOT, help="Dataset root folder")
    parser.add_argument("--dataset-subset", type=str, default=DATASET_SUBSET, help="Dataset subset (A-G)")
    parser.add_argument(
        "--dataset-split",
        type=str,
        default=DATASET_SPLIT,
        choices=["Training", "Validation", "Test"],
        help="Dataset split",
    )
    parser.add_argument("--dataset-instance", type=str, default=DATASET_INSTANCE, help="Instance folder name")
    parser.add_argument(
        "--skip-problem-print",
        action="store_true",
        help="Skip writing the full problem definition to data/results.txt",
    )
    return parser.parse_args()


def build_env(args):
    return BusSchedulingEnv(
        data_source=args.data_source,
        dataset_root=args.dataset_root,
        dataset_subset=args.dataset_subset,
        dataset_split=args.dataset_split,
        dataset_instance=args.dataset_instance,
    )


def train(args):
    """
    Trains the PPO agent.

    Returns:
        tuple: (episode_rewards, avg_rewards, trained_agent, env)
    """
    create_results_directory()

    start_time = time.time()
    print(f"Starting PPO training at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    with open("data/ppo_training.txt", "w", encoding="utf-8") as handle:
        handle.write("")

    env = build_env(args)
    if not args.skip_problem_print:
        env.print_problem()

    agent = PPOAgent(env.observation_space_dim, env.action_space_dim, hidden_size=HIDDEN_SIZE)
    episode_rewards = []
    avg_rewards = []

    for episode in range(args.episodes):
        state = env.reset()
        done = False
        trajectory = []
        total_reward = 0.0

        while not done:
            valid_actions = env.get_valid_actions()
            action, log_prob, _ = agent.policy.act(state, valid_actions=valid_actions)
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, log_prob, reward, done, list(valid_actions)))
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        agent.update(trajectory)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_rewards.append(avg_reward)
            progress = f"Episode {episode + 1}/{args.episodes}, Average Reward: {avg_reward:.2f}\n"
            print(progress, end="")
            with open("data/ppo_training.txt", "a", encoding="utf-8") as handle:
                handle.write(progress)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"PPO training completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total PPO training time: {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")

    with open("data/ppo_training.txt", "a", encoding="utf-8") as handle:
        handle.write(f"\nTraining Time: {training_time:.2f} seconds ({training_time / 60:.2f} minutes)\n")

    return episode_rewards, avg_rewards, agent, env


def evaluate(agent, env, num_episodes=10):
    """
    Evaluates the trained PPO agent.
    """
    eval_rewards = []
    total_buses_used = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            valid_actions = env.get_valid_actions()
            action, _, _ = agent.policy.act(state, valid_actions=valid_actions)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        eval_rewards.append(episode_reward)
        total_buses_used.append(env.get_total_buses_used())

    results = (
        "PPO Evaluation Results:\n"
        f"Average Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}\n"
        f"Average Buses Used: {np.mean(total_buses_used):.2f} ± {np.std(total_buses_used):.2f}\n"
        f"Best Episode Reward: {max(eval_rewards):.2f}\n"
        f"Minimum Buses Used: {min(total_buses_used)}\n"
    )
    write_results(results, "results.txt")
    env.print_solution("data/results.txt")


if __name__ == "__main__":
    cli_args = parse_args()
    rewards, avg_rewards, agent, env = train(cli_args)
    plot_rewards(rewards, "ppo_training_rewards.png", "PPO Training Progress")
    evaluate(agent, env, num_episodes=cli_args.eval_episodes)
