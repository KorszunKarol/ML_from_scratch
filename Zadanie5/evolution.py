from deap import base, creator, tools, algorithms
import random
import numpy as np
from neural_net import NeuralNetwork
from data_generator import DataGenerator
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


def load_weights(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)



class EvolutionaryNeuralOptimizer:
    def __init__(
        self,
        neural_network,
        population_size=20,
        generations=10,
        mutation_rate=0.1,
        mutation_scale=0.1,
    ):
        self.nn = neural_network
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, -1.0, 1.0)
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_float,
            len(self.nn.get_params()),
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mutate", self.mutate)

    def evaluate(self, individual):
        self.nn.set_params(individual)
        predictions = self.nn.feedforward(self.nn.X)
        loss = self.nn.mse(self.nn.y, predictions)
        return (loss,)

    def mutate(self, individual):
        mutated_individual = creator.Individual(individual)
        for i in range(len(mutated_individual)):
            if random.random() < self.mutation_rate:
                mutated_individual[i] += np.random.normal(0, self.mutation_scale)
        return mutated_individual

    def run(self):

        try:
            best_weights = load_weights('best_weights_es.pkl')
            population = [best_weights]
        except Exception as e:
            population = self.toolbox.population(n=self.population_size)

        for gen in range(self.generations):
            individual = population[0]
            offspring = self.mutate(individual)
            individual.fitness.values = self.evaluate(individual)
            offspring.fitness.values = self.evaluate(offspring)
            print(f"Loss: {population[0].fitness.values[0]} at generation {gen}")
            if offspring.fitness.values[0] <= individual.fitness.values[0]:
                population[0] = offspring
                self.save_weights(population[0])

        return population

    def save_weights(self, weights):
        with open('best_weights_es.pkl', 'wb') as f:
            pickle.dump(weights, f)

    def plot_predictions(self):
        predictions = self.nn.feedforward(self.nn.X)
        dg = DataGenerator()
        dg.plot_data(self.nn.X, self.nn.y, predictions=predictions)


def main():

    X = np.linspace(-10, 10, 1000).reshape(-1, 1)
    y = DataGenerator.eval_func(X)

    nn = NeuralNetwork(
        X,
        y,
        neurons=[200, 100, 50, 25],
        learning_rate=0.01,
        clip_value=1.0,
    )

    ev_optimizer = EvolutionaryNeuralOptimizer(
        nn,
        population_size=1,
        generations=5_000,
        mutation_scale=0.0003,
    )
    ev_optimizer.run()
    best_weights = load_weights('best_weights_es.pkl')

    nn.set_params(best_weights)

    X_range = np.linspace(-10, 10, 1000).reshape(-1, 1)
    y_pred = nn.predict(X_range)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=X_range.squeeze(), y=y_pred.squeeze(), label="Predicted")
    sns.lineplot(x=X.squeeze(), y=DataGenerator.eval_func(X).squeeze(), label="True")
    plt.title("Comparison of predicted and true function")
    plt.xlabel("X")
    plt.ylabel("y")

    plt.legend()
    plt.grid(True)
    plt.savefig('evolutionary_strategy.png', dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
