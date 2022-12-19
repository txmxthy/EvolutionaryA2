# Dependencies
import os
from operator import add
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local Imports
from Configuration.tools import configure_comparison
from src.common import utils
from src.common.utils import print_header
from src.p1.Algorithms.base import round_metrics


def main():
    config, metrics = configure_comparison()
    metrics = run_comparison(config, metrics, 20)
    metrics = run_comparison(config, metrics, 50)


def run_comparison(conf, metrics, d=20):
    """
       run_comparison(conf, metrics, d=20)

       Run the comparison for each problem and approach with a specified number of dimensions.

       Parameters
       ----------
       conf : dict
           The configuration dictionary.
       metrics : dict
           The metrics dictionary.
       d : int
           The number of dimensions.

       Returns
       -------
       dict
           The updated metrics dictionary.

       Examples
       --------
       """
    print("\n")
    print_header(f"Running comparison with dim {d}")

    # Compare each problem
    for problem in conf['problems']:
        problem_name = problem.__name__
        print("\n")
        print_header("Solving: {}".format(problem_name), sep="-")

        # Run the comparison for the selected problem with each approach
        for approach in conf['approaches']:
            approach_name = approach.__name__
            print("\n")
            print("With approach_name: {}".format(approach_name))
            convergence = np.array([0 for _ in range(conf['generations'])], dtype=float)
            fitness_values = []
            print("\n")
            for _ in tqdm(range(1, conf['iterations'] + 1), ascii=True, desc="\t\tRunning iterations"):
                pop, conv = approach(function=problem,
                                     generations=conf['generations'],
                                     population_size=conf['population'],
                                     dim=d)
                fitness_values.append(pop[0].fitness)
                convergence = list(map(add, conv, convergence))

            metrics = plot_convergence(d, approach_name, convergence, problem_name, fitness_values, metrics,
                                       conf['generations'])
    print_results(metrics, d)
    return metrics


def plot_convergence(d, approach_name, convergence, problem_name, fitness_values, metrics, generations):
    """
        plot_convergence(dim, approach_name, convergence, function, fitness_values, metrics, generations)

        Plot the convergence of the algorithm.

        Parameters
        ----------
        d : int
            The dimension of the problem.
        approach_name : str
            The name of the approach.
        convergence : list
            The convergence of the algorithm.
        problem_name : str
            The name of the function.
        fitness_values : list
            The fitness values of the algorithm.
        metrics : dict
            The metrics of the algorithm.
        generations : int
            The number of generations.

        Returns
        -------
        dict
            The metrics of the algorithm.

        """
    metrics[problem_name + "_" + approach_name] = round_metrics(fitness_values)
    plt.plot(range(0, generations), np.mean([convergence], axis=0))
    plt.xlabel("Generation")
    plt.ylabel('Fitness')
    plt.title(f"{problem_name} {approach_name}  (with D as {d})")
    utils.render(text=os.path.basename(__file__),
                 name=f"{problem_name}_{approach_name}_D{d}", plot=plt)
    plt.show()
    return metrics


def print_results(metrics, d):
    """
        print_results(metrics, d)

        Print the results of the  models.

        Parameters
        ----------
        metrics : dict
            The metrics of the  models.
        d : int
            The number of features.

        Returns
        -------
        None
    """
    results = pd.DataFrame(metrics, index=list(metrics.values())[0].keys(), columns=metrics.keys())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print_header(f"Results with dim {d}")
        print(results)


if __name__ == "__main__":
    main()
