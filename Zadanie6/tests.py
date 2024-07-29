from taxi import q_learning
import gym
import json
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def test_strategies(
    env,
    num_episodes,
    alpha,
    gamma,
    epsilon,
    epsilon_decay,
    strategies,
    num_simulations,
    epsilon_first_episodes=0,
):
    rewards = {strategy: np.zeros(num_episodes) for strategy in strategies}
    for _ in range(num_simulations):
        for strategy in strategies:
            _, total_rewards = q_learning(
                env,
                num_episodes,
                alpha,
                gamma,
                epsilon,
                epsilon_decay,
                strategy,
                epsilon_first_episodes,
            )
            rewards[strategy] += total_rewards

    for strategy in strategies:
        rewards[strategy] /= num_simulations

    sns.set_style(style="whitegrid")
    for strategy, total_rewards in rewards.items():
        total_rewards_pd = pd.Series(total_rewards).rolling(window=10).mean()
        plt.plot(total_rewards_pd, label=strategy)
    plt.title(
        "Total Reward per Episode for Different Strategies"
    )
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (Moving Average)")
    plt.legend()
    plt.savefig("strategies.png", dpi=600)
    plt.show()

    return rewards


def test_gamma_values(
    env, num_episodes, alpha, epsilon, gamma_values, epsilon_decay, plot=False, n=10
):
    results = {}

    for gamma in gamma_values:
        total_rewards_all_runs = []
        for _ in range(n):
            Q, total_rewards = q_learning(
                env, num_episodes, alpha, gamma, epsilon, epsilon_decay
            )
            total_rewards_all_runs.append(total_rewards)

        avg_total_rewards = np.mean(total_rewards_all_runs, axis=0)

        Q_dict = {k: v.tolist() for k, v in Q.items()}
        key = f"alpha={alpha}, epsilon={epsilon}, gamma={gamma}"
        results[key] = {"Q": Q_dict, "total_rewards": avg_total_rewards}

    if plot:
        sns.set_theme(style="whitegrid")
        window_size = 10

        plt.figure(figsize=(10, 6))

        for key, result in results.items():
            rewards_smoothed = (
                pd.Series(result["total_rewards"])
                .rolling(window_size, min_periods=window_size)
                .mean()
            )
            sns.lineplot(data=rewards_smoothed, label=key)

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")

        plt.legend()
        plt.savefig("rewards_2.png", dpi=600)
        plt.show()

    return results


def get_epsilon_values(env, num_episodes, alpha, gamma, epsilon_values, epsilon_decay):
    results = {}

    for epsilon in epsilon_values:
        Q, total_rewards = q_learning(
            env, num_episodes, alpha, gamma, epsilon, epsilon_decay
        )
        Q_dict = {k: v.tolist() for k, v in Q.items()}
        key = f"alpha={alpha}, epsilon={epsilon}, epsilon_decay={epsilon_decay}"
        results[key] = {"Q": Q_dict, "total_rewards": total_rewards}

    return results


def test_epsilon_values(env, num_episodes, alpha, gamma, epsilon_values, epsilon_decay):
    env = gym.make("Taxi-v3")
    num_episodes = 300
    alpha = 0.9
    epsilon_decay = 0.9
    gamma = 0.99
    epsilon_values = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

    num_runs = 20
    all_results = []

    for _ in range(num_runs):
        results = get_epsilon_values(
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
    return avg_results


def plot_q_values(Q, env):
    Q_array = np.zeros((env.observation_space.n, env.action_space.n))

    for state, actions in Q.items():
        Q_array[state] = actions

    Q_array = Q_array - np.min(Q_array)
    if np.max(Q_array) != 0:
        Q_array = Q_array / np.max(Q_array)

    plt.figure(figsize=(10, 6))
    sns.heatmap(Q_array, cmap="viridis", cbar_kws={"label": "Q-value"}, cbar=True)
    plt.xlabel("Action")
    plt.ylabel("State")
    plt.title("Q-values for each state and action")
    plt.show()


def plot_policy(Q, env):
    policy = np.array([np.argmax(actions) for actions in Q.values()])
    policy_smooth = moving_average(policy, n=20)

    plt.figure(figsize=(10, 6))
    plt.plot(policy_smooth)
    plt.xlabel("State")
    plt.ylabel("Best Action")
    plt.title("Policy for each state (Smoothed)")
    plt.show()


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_value_function(Q, env):
    value_function = np.array([np.max(actions) for actions in Q.values()])
    value_function_smooth = moving_average(value_function, n=50)

    plt.figure(figsize=(10, 6))
    plt.plot(value_function_smooth)
    plt.xlabel("State")
    plt.ylabel("Value")
    plt.title("Value function for each state (Smoothed)")
    plt.show()


def main():
    env = gym.make("Taxi-v3")
    num_episodes = 300
    alpha = 0.9
    epsilon_decay = 0.9
    gamma = 0.99
    gamma_values = [0.1, 0.5, 0.7, 0.99]
    epsilon_values = [0.1, 0.5, 0.7, 0.99]
    epsilon = 0.7
    # test_epsilon_values(env, num_episodes, alpha, gamma, epsilon_values, epsilon_decay)
    # test_gamma_values(
    #     env, num_episodes, alpha, epsilon, gamma_values, epsilon_decay, plot=True, n=50
    # )
    test_strategies(
        env,
        num_episodes,
        alpha,
        gamma,
        epsilon,
        epsilon_decay,
        strategies=["epsilon_greedy", "epsilon_first"],
        epsilon_first_episodes=15,
        num_simulations=100,
    )


if __name__ == "__main__":
    main()
