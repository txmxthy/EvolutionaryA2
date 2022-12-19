import copy
import random
import numpy as np

from src.p1.Algorithms.base import fitness, tournament_selection, select, BaseIndividual


def EvolutionaryProgramming(generations, function, population_size=80, dim=20, mutation_p=0.5,
                            bounds=(-30, 30)):
    """
       EvolutionaryProgramming(generations, function, population_size=80, dim=20, mutation_p=0.5,
                               bounds=(-30, 30))

       Perform evolutionary programming on a function.

       Parameters
       ----------
       generations : int
           The number of generations to evolve for.
       function : function
           The function to be optimised.
       population_size : int
           The size of the population.
       dim : int
           The dimensionality of the problem.
       mutation_p : float
           The probability of mutation.
       bounds : tuple
           The bounds of the problem.

       Returns
       -------
       list
           The final population.
       numpy.ndarray
           The convergence of the best individual.
       """
    # Create population
    population = [BaseIndividual(dim, bounds) for _ in range(population_size)]
    suitability = [fitness(ind, function) for ind in population]
    for ind, fit in zip(population, suitability):
        ind.fitness = fit

    convergence = np.array([np.inf for _ in range(generations)])

    # Begin the evolution
    for gen in range(generations):
        percent_best = 0.1
        # Take the best 10% of the population
        best = select(population, int(population_size * percent_best))
        offspring = list(map(copy.deepcopy, population))
        # Apply mutation probability to the rest of the population
        for index, mutant in enumerate(offspring):

            if random.random() < mutation_p:
                mutant = mutant.mutate(mutation_p, 1, 0.5)
                mutant.fitness = fitness(mutant, function)
                offspring[index] = mutant

        population[:] = tournament_selection(best + population + offspring, population_size, 0.25)
        convergence[gen] = population[0].fitness
    return population, convergence
