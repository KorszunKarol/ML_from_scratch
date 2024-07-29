import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import pandas as pd
import seaborn as sns
import json
from typing import Dict, List, Tuple, Any

env = gym.make("Taxi-v3")


logging.basicConfig(level=logging.INFO)


def q_learning(
    env: gym.Env,
    num_episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_decay: float,
    exploration_strategy: str = "epsilon_greedy",
    epsilon_first_episodes: int = 0,
) -> Tuple[Dict, List[int]]:

    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if exploration_strategy == "epsilon_greedy":
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[state])
            elif exploration_strategy == "epsilon_first":
                if episode < epsilon_first_episodes:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[state])
            else:
                raise ValueError(
                    f"Unknown exploration strategy: {exploration_strategy}"
                )

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            state = next_state

        total_rewards.append(total_reward)
        epsilon = max(epsilon * epsilon_decay, 0.01)

        if episode % 100 == 0:
            logging.info(f"Episode: {episode}, Total Reward: {total_reward}")

    return Q, total_rewards


def test(Q: Dict, env: gym.Env) -> None:
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        env.render()


def plot_results(results: Dict[str, Any], window_size: int = 100) -> None:

    sns.set_theme(style="whitegrid", palette="husl")
    plt.figure(figsize=(12, 8))

    line_styles = ["-", "--", ":", "-."]
    style_index = 0

    for key, value in results.items():
        alpha, epsilon_decay = key.split(", ")
        alpha = float(alpha.replace("alpha=", ""))
        epsilon_decay = float(epsilon_decay.replace("epsilon_decay=", ""))

        rewards_smoothed = (
            pd.Series(value["total_rewards"])
            .rolling(window_size, min_periods=window_size)
            .mean()
        )
        sns.lineplot(
            x=range(len(rewards_smoothed)),
            y=rewards_smoothed,
            label=f"alpha={alpha}, epsilon_decay={epsilon_decay}",
            linestyle=line_styles[style_index % len(line_styles)],
        )

        style_index += 1

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-learning: Impact of Hyperparameters on Total Rewards")
    plt.legend()
    plt.savefig("decay_alpha.png", dpi=600)
    plt.show()


def test_epsilon_values(
    env: gym.Env
    num_episodes: int,
    alpha: float,
    gamma: float,
    epsilon_values: List[float],
    epsilon_decay: float,
) -> Dict[str, Any]:
    results = {}

    for epsilon in epsilon_values:
        Q, total_rewards = q_learning(
            env, num_episodes, alpha, gamma, epsilon, epsilon_decay
        )
        Q_dict = {k: v.tolist() for k, v in Q.items()}
        key = f"alpha={alpha}, epsilon={epsilon}, epsilon_decay={epsilon_decay}"
        results[key] = {"Q": Q_dict, "total_rewards": total_rewards}

    return results


def main():
    # num_episodes = 800
    # gamma = 0.99

    # alphas = [0.1, 0.5, 0.9]
    # epsilon_decays = [0.9, 0.95, 0.99]

    # results = {}

    # for alpha in alphas:
    #     for epsilon_decay in epsilon_decays:
    #         epsilon = 1.0
    #         Q, total_rewards = q_learning(
    #             env, num_episodes, alpha, gamma, epsilon, epsilon_decay
    #         )
    #         # Convert the defaultdict of ndarrays to a dict of lists
    #         Q_dict = {k: v.tolist() for k, v in Q.items()}
    #         key = f"alpha={alpha}, epsilon_decay={epsilon_decay}"
    #         results[key] = {'Q': Q_dict, 'total_rewards': total_rewards}

    # json.dump(results, open("results.json", "w"))

    # # print("hey")

    # results = json.load(open("results.json", "r"))
    # print(type(results))
    # print(type(next(iter(results.values()))))
    # best_key = max(results, key=lambda k: sum(results[k]["total_rewards"]))
    # best_alpha, best_epsilon_decay = map(
    #     float, best_key.replace("alpha=", "").replace(" epsilon_decay=", "").split(",")
    # )
    # best_Q = results[best_key]["Q"]
    # print(best_alpha, best_epsilon_decay)
    # plot_results(results)

    # test(best_Q, env)
    env = gym.make("Taxi-v3")
    num_episodes = 300
    alpha = 0.9
    epsilon_decay = 0.9
    gamma = 0.99
    epsilon_values = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

    num_runs = 20
    all_results = []

    for _ in range(num_runs):
        results = test_epsilon_values(
            env, num_episodes, alpha, gamma, epsilon_values, epsilon_decay
        )
        all_results.append(results)

    avg_results = {}
    for key in all_results[0].keys():
        avg_results[key] = {
            "total_rewards": np.mean(
                [run[key]["total_rewards"] for run in all_results], axis=0
            )
        }

    sns.set_theme(style="whitegrid")
    window_size = 10

    plt.figure(figsize=(10, 6))

    line_styles = ["-", "--", ":", "-."]
    style_index = 0

    for key, result in avg_results.items():
        rewards_smoothed = (
            pd.Series(result["total_rewards"])
            .rolling(window_size, min_periods=window_size)
            .mean()
        )
        sns.lineplot(
            data=rewards_smoothed,
            label=key,
            linestyle=line_styles[style_index % len(line_styles)],
        )
        style_index += 1

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig("rewards.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
