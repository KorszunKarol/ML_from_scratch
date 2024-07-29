import autograd.numpy as np
from autograd import grad
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import time
import random
import matplotlib
import pandas as pd
from functools import partial


def plot_q_3D(x_vals, y_vals, alpha=1, n=10):
    x = np.linspace(-100, 100, 10)
    X, Y = np.meshgrid(x, x, indexing="ij", sparse=True)
    Z = np.array([[q(x0, alpha, n) for x0 in row] for row in X])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.5)
    ax.scatter(x_vals, list(range(len(x_vals))), y_vals, color="red")

    # ax.set_ylim(0, 100)
    ax.set_xlabel("X")
    ax.set_ylabel("Iteration")
    ax.set_zlabel("Function value")
    plt.title("3D plot of the function q(x) and the trajectory of convergence")
    plt.show()


def q(x, alpha, n):
    x = np.array(x)
    assert len(x) == n, f"Expected x to be of length {n}, but got {len(x)}"
    i = np.arange(1, n + 1)
    sum = np.sum((alpha ** ((i - 1) / (n - 1))) * (x[i - 1] ** 2))
    return sum


def plot_q(x_vals, y_vals, name, alpha=1, n=10):
    x = np.linspace(-100, 100, 100)
    x = np.array([x for _ in range(n)])
    y = np.array([[q(x0, alpha, n) for x0 in row] for row in x])

    for i in range(n):
        plt.plot(x[i], y[i])

    plt.plot(x_vals, y_vals, color="blue", linestyle="--", marker="o")
    plt.scatter(x_vals[-1], y_vals[-1], color="red")
    plt.xlabel("x")
    plt.ylabel("q(x)")
    plt.title("Plot of the function q(x)")
    plt.grid(True)

    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )
    plt.savefig(name)


def test_gradient_descent(learning_rates, alphas, x0, q, max_iterations, n=10):
    results = []
    learning_rates = np.linspace(0.000001, 0.01, num=1000)
    for learning_rate in learning_rates:
        for alpha in alphas:
            start_time = time.time()
            partial_q = partial(q, alpha=alpha, n=n)
            gradient_descent(partial_q, x0, learning_rate, max_iterations, alpha, n)
            end_time = time.time()

            results.append(
                {
                    "learning_rate": learning_rate,
                    "alpha": alpha,
                    "execution_time": end_time - start_time,
                }
            )
    df = pd.DataFrame(results)
    sns.lineplot(
        data=df,
        x="learning_rate",
        y="execution_time",
        hue="alpha",
        palette=["red", "green", "blue"],
    )
    plt.yscale("log")
    plt.show()
    return results


def plot_learning_rate(learning_rates, alphas, x0, q, num_iterations, n=10):
    fig, axs = plt.subplots(len(alphas), 1, figsize=(10, 5 * len(alphas)))

    for i, alpha in enumerate(alphas):
        for learning_rate in learning_rates:
            q_partial = partial(q, alpha=alpha, n=n)
            _, y_vals = gradient_descent(
                q_partial, x0, learning_rate, num_iterations, alpha=alpha, n=n
            )
            axs[i].plot(y_vals, label=f"learning rate = {learning_rate}")
        axs[i].set_xlabel("Iteration")
        axs[i].set_ylabel("Function value")
        axs[i].set_title(f"Plot of the function q(x) for alpha = {alpha}")
        axs[i].legend()

    plt.tight_layout()
    plt.show()
    return


def gradient_descent(
    func, x0, learning_rate=0.001, max_iterations=1000, tolerance=1e-4, alpha=1, n=10
):
    x = x0
    x_prev = 0
    gradient_func = grad(func)
    x_values = []
    func_values = []
    for _ in range(max_iterations):
        gradient = gradient_func(x)
        x = x - learning_rate * gradient
        x_values.append(x)
        func_value = func(x)
        func_values.append(func_value)
        if np.linalg.norm(x - x_prev) < tolerance:
            break
        x_prev = x
    return x_values, func_values


def main():
    x0 = np.array([random.uniform(-100, 100) for _ in range(10)])
    alphas = [1, 10, 100]
    learning_rates = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    n = 10
    q_partial = partial(q, alpha=alphas[2], n=n)
    x, y = gradient_descent(
        q_partial,
        x0,
        learning_rate=0.001,
        max_iterations=10000,
        alpha=alphas[2],
        n=n,
        tolerance=1e-7,
    )
    print(f"{x[-1]} \n {y[-1]}")
    return


if __name__ == "__main__":
    main()
