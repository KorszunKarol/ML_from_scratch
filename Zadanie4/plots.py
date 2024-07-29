import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from mpl_toolkits.mplot3d import Axes3D


def plot_accuracy_line():
    with open('params.json', 'r') as f:
        data = json.load(f)

    data = [(float(k.split(", ")[0][1:]), float(k.split(", ")[1][:-1]), v) for k, v in data.items()]
    df = pd.DataFrame(data, columns=['learning_rate', 'lambda_param', 'accuracy'])

    lambda_params = df['lambda_param'].unique()

    for param in lambda_params:
        subset = df[df['lambda_param'] == param]
        plt.plot(subset['learning_rate'], subset['accuracy'], label=f'lambda={param}')

    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def plot_accuracy_3d():
    with open('params.json', 'r') as f:
        data = json.load(f)

    data = [(float(k.split(", ")[0][1:]), float(k.split(", ")[1][:-1]), v) for k, v in data.items()]
    df = pd.DataFrame(data, columns=['learning_rate', 'lambda_param', 'accuracy'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df['learning_rate'], df['lambda_param'], df['accuracy'], c='r', marker='o')

    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Lambda Param')
    ax.set_zlabel('Accuracy')

    plt.show()

def plot_accuracy_heatmap(filename: str):
    with open(filename, 'r') as f:
        data = json.load(f)

    data = [(float(k.split(", ")[0][1:]), float(k.split(", ")[1][:-1]), v) for k, v in data.items()]
    df = pd.DataFrame(data, columns=['gamma', 'degree', 'accuracy'])

    pivot_table = df.pivot(index='gamma', columns='degree', values='accuracy')
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")

    plt.savefig('plots/heatmap_poly.png', dpi=600)
    plt.show()

def main():
    plot_accuracy_heatmap(filename='params_poly.json')

if __name__ == '__main__':
    main()