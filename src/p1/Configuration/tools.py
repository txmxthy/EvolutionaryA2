import numpy as np
from src.p1.Algorithms.DE import DifferentialEvolution
from src.p1.Algorithms.EP import EvolutionaryProgramming
from src.p1.Algorithms.base import round_metrics


def Rosenbrock(x):
    return sum([100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
                for i in range(len(x) - 1)])


def Griewanks(x):
    return 1 + 1 / 4000 * sum([x[i] ** 2
                               for i in range(len(x))]) - np.prod([np.cos(x[i] / np.sqrt(i + 1))
                                                                   for i in range(len(x))])


def configure_comparison():
    """
    Configure the comparison between the two problems
    :return: dict
    """

    # Defaults
    # Chosen as a multiple of 3 for mutation,
    # Divisible by two for crossover
    # Approx 1/3 of the population for tournament selection
    iterations = 24
    generations = 48
    population = 96
    problems = (Rosenbrock, Griewanks)
    approaches = (EvolutionaryProgramming, DifferentialEvolution)

    # Get the parameters from user input, if enter is pressed, use the default value
    # if user wants to continue with default values, just press enter
    custom = input("Use custom parameters? (y/n): ")
    if custom == "y":
        iterations = int(input("Number of iterations: "))
        generations = int(input("Number of generations: "))
        population = int(input("Population size: "))
        # Select problems from available
        print("Select problems to compare:")
        for i, f in enumerate(problems):
            print(f"{i + 1}. {f.__name__}")
        selected = input("Enter the numbers of the problems separated by a space: ")
        selected = [int(i) for i in selected.split()]
        problems = [problems[i - 1] for i in selected]
        # Select approaches from available
        print("Select approaches to compare:")
        for i, f in enumerate(approaches):
            print(f"{i + 1}. {f.__name__}")
        selected = input("Enter the numbers of the approaches separated by a space: ")
        selected = [int(i) for i in selected.split()]
        approaches = [approaches[i - 1] for i in selected]

    config = {
        "iterations": iterations,
        "generations": generations,
        "population": population,
        "problems": problems,
        "approaches": approaches,
    }
    return config, {}
