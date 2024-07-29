import numpy as np
from autograd import grad, elementwise_grad
import matplotlib.pyplot as plt
import time
import random
from cec2017.functions import f3, f19, f10
from typing import Callable, Tuple, Optional, Dict, List
from matplotlib.axes import Axes
import sys
import os
import pandas as pd
from scipy.optimize import minimize
import seaborn as sns

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from Zadanie1.gradient_descent import gradient_descent


def visualize_functions(function, domain=(-100, 100), points=30, dimension=2, ax=None):
    xys = np.linspace(domain[0], domain[1], points)
    xys = np.transpose([np.tile(xys, len(xys)), np.repeat(xys, len(xys))])

    if dimension > 2:
        tail = np.zeros((xys.shape[0], dimension - 2))
        x = np.concatenate([xys, tail], axis=1)
        zs = function(x)
    else:
        zs = function(xys)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    X = xys[:, 0].reshape((points, points))
    Y = xys[:, 1].reshape((points, points))
    Z = zs.reshape((points, points))

    surf = ax.plot_surface(X, Y, Z, cmap="plasma")
    ax.contour(X, Y, Z, zdir="z", cmap="plasma", linestyles="solid", offset=40)
    ax.contour(X, Y, Z, zdir="z", colors="k", linestyles="solid")
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.15, label="")
    ax.set_title(function.__name__, fontsize=20)

    ax.contour(X, Y, Z, levels=10, cmap="plasma", linestyles="solid", offset=40)

    ax.set_title(function.__name__, fontsize=20)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.set_zlabel("y", fontsize=14)

    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    plt.show()

    return fig


def ES_1_plus_1(
    f: Callable,
    optim_params: Dict[str, float],
):
    dimension = optim_params.get("dimension")
    generations = optim_params.get("generations")
    domain_lower_bound = optim_params.get("domain_lower_bound")
    domain_upper_bound = optim_params.get("domain_upper_bound")
    n = optim_params.get("n")
    parent = np.random.uniform(
        domain_lower_bound, domain_upper_bound, size=(1, dimension)
    )
    alpha_1 = optim_params.get("alpha_1")
    alpha_2 = optim_params.get("alpha_2")
    recent_successes = np.zeros(n)
    sigma = optim_params.get("sigma")
    y_vals = []

    for i in range(generations):
        child = parent + np.random.normal(0, 1, dimension) * sigma

        if f(child) < f(parent):
            parent = child
            recent_successes[i % n] = 1
        else:
            recent_successes[i % n] = 0

        if i % n == 0:
            success_freq = np.sum(recent_successes) / n
            if success_freq < 0.2:
                sigma = sigma * alpha_1
            else:
                sigma = sigma * alpha_2
            recent_successes = np.zeros(n)
        y_vals.append(f(parent)[0])
    return parent, f(parent), y_vals


def run_evolution_strategy(optim_params):
    y_list = []
    sigmas = np.arange(0.0, 10.5, 0.5)
    for sigma in sigmas:
        y_temp = []
        optim_params["sigma"] = sigma
        for i in range(50):
            x, y = ES_1_plus_1(f=f19, optim_params=optim_params)
            print("hey")
            y_temp.append(y)
        print(sigma)
        print(np.mean(y_temp))
        y_list.append(np.mean(y_temp))

    df = pd.DataFrame({"sigma": sigmas, "average_result": y_list})

    df.to_csv(f"results_f19.csv", index=False)

    return y_list


def f19_wrapper(x):
    x_reshaped = x.reshape(1, -1)
    return f19(x_reshaped)[0]


def f3_wrapper(x):
    x_reshaped = x.reshape(1, -1)
    return f3(x_reshaped)[0]


def plot_convergence(
    es_values: pd.DataFrame,
    gd_values: pd.DataFrame,
    param_3,
    param_4,
    param_5,
    param_6,
    param_7,
) -> None:
    print(len(es_values), len(gd_values))
    if len(es_values) != len(gd_values):
        diff = len(gd_values) - len(es_values)
        if len(es_values) < len(gd_values):
            es_values.extend([es_values[-1]] * diff)
        else:
            gd_values.extend([gd_values[-1]] * -diff)
        print(len(es_values), len(gd_values))

    data = pd.DataFrame(
        {
            "1^(-0.25), 1": es_values,
            "2^(-0.25), 2": gd_values,
            "3^(-0.25), 3": param_3,
            "4^(-0.25), 4": param_4,
            "5^(-0.25), 5": param_5,
            "6^(-0.25), 6": param_6,
            "0.82, 1.22": param_7,
        }
    )

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    palette = sns.color_palette("hsv", len(data.columns))

    sns.lineplot(data=data, palette=palette, linewidth=2.5, dashes=True, alpha=0.5)

    plt.title("Convergence of f3  depending on alpha parameter", fontsize=30)
    plt.xlabel("Iteration", fontsize=24)
    plt.ylabel("Function Value", fontsize=24)
    plt.tick_params(labelsize=20)
    plt.yscale("log")
    plt.legend(title="alpha", title_fontsize="23", fontsize=22)
    plt.grid(True)
    plt.show()


def compare_alphas() -> None:
    pass


def main() -> None:
    optim_params: Dict[str, float] = {
        "dimension": 10,
        "generations": 10_000,
        "domain_lower_bound": -400,
        "domain_upper_bound": 400,
        "n": 10,
        "alpha_1": 2 ** (-0.25),
        "alpha_2": 2,
        "sigma": 3,
    }
    plot_convergence(
        pd.read_csv("csv/results_f3_alpha_1.0.csv")["average_result"],
        pd.read_csv("csv/results_f3_alpha_2.0.csv")["average_result"],
        pd.read_csv("csv/results_f3_alpha_3.0.csv")["average_result"],
        pd.read_csv("csv/results_f3_alpha_4.0.csv")["average_result"],
        pd.read_csv("csv/results_f3_alpha_5.0.csv")["average_result"],
        pd.read_csv("csv/results_f3_alpha_6.0.csv")["average_result"],
        pd.read_csv("csv/results_f3_alpha_book.csv")["average_result"],
    )

    return


if __name__ == "__main__":
    main()
