import array
import random
from operator import add
import matplotlib.pyplot as plt
import numpy
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from Algorithms.PBIL import ProbabilityBasedIncrementalLearning
from src.common import utils
from src.common.utils import load_knapsack, print_header


def Evaluation(individual):
    a = 50
    w = 0.0
    v = 0.0
    for i in ITEMS.itertuples():
        if individual[i.Index]:
            w += i.weight
            v += i.value
    penalty = a * max(0, w - W_MAX)
    return v, penalty,


def make_toolbox(PBIL):
    toolbox = base.Toolbox()
    creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", array.array, typecode='b', fitness=creator.Fitness)
    toolbox.register("evaluate", Evaluation)
    toolbox.register("generate", PBIL.generate, creator.Individual)
    toolbox.register("update", PBIL.update)
    return toolbox


def main():
    global ITEMS, I_MAX, W_MAX
    knapsack_data = load_knapsack()
    generations = 100
    solutions = [1514, 295, 9767]

    for i, (name, dataset) in enumerate(knapsack_data.items()):
        convergence = [0] * generations
        suitability = []
        # Set seeds
        for seed in [1, 42, 1337, 12345, 123456]:
            random.seed(seed)
            np.random.seed(seed)

            # Set global configuration for the problem
            ITEMS = dataset["Items"]
            I_MAX = dataset["Count"]
            W_MAX = dataset["Limit"]

            # Create the PBIL algorithm
            EDA = ProbabilityBasedIncrementalLearning(dim=dataset.get("Count"),
                                                      learning_rate=0.7,
                                                      mut_prob=0.1,
                                                      mut_shift=0.05,
                                                      pop_sizer=100)
            # Make Toolbox and stats
            toolbox = make_toolbox(EDA)
            stats = make_stats()

            # Run
            pop, logbook = algorithms.eaGenerateUpdate(toolbox, generations, stats=stats, verbose=False)
            convergence = list(map(add, [abs(solutions[i] - fit[0]) for fit in logbook.select("max")], convergence))
            suitability.append(pop[0].fitness.values[0])

        display_results(convergence, suitability, generations, name)


def make_stats():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    return stats


def display_results(convergence, suitability, generations, name):
    """
    Display the results of the algorithm
    """
    # Print
    print_header("Results for {}".format(name))
    print("Fitnesses: {}".format(suitability))

    print("Mean:", np.mean(suitability))
    print("Std:", np.std(suitability))
    # Plot
    plt.plot(range(generations), np.mean([convergence], axis=0), label=name)
    plt.title(name)
    plt.xlabel("Generation")
    plt.ylabel("Value (Dist to Optimal Value)")
    utils.render(plt, "part2", "knapsack" + name)
    plt.show()


if __name__ == "__main__":
    main()
