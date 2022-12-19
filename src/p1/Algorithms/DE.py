import copy
import random
import numpy as np
from Algorithms.base import BaseIndividual
from src.p1.Algorithms.base import fitness


def DifferentialEvolution(generations, function, population_size=100, dim=20, mutation_p=0.5,
                          bounds=(-30, 30)):
    # Create Population
    population = [DeIndividual(dim, bounds) for i in range(population_size)]
    suitability = [fitness(ind, function) for ind in population]
    convergence = np.array([np.inf for _ in range(generations)])

    for ind, fit in zip(population, suitability):
        ind.f = fit

    # Start the evolution
    for generation in range(generations):

        offspring = list(map(copy.deepcopy, population))
        # Keep mutation if it is more suitable than parent
        for i, mutant in enumerate(offspring):

            # Take three random individuals from the population
            a, b, c = random.sample(population, 3)
            while not (mutant != a != b != c != mutant):
                a, b, c = random.sample(population, 3)

            # Roll for mutation
            if random.random() < mutation_p:
                mutant = mutant.mutate(mutation_p, a, b)

            # Apply Cross over
            crossover = mutant.crossover(c)
            crossover.fitness = fitness(crossover, function)

            if crossover.fitness < population[i].fitness:
                offspring[i] = crossover

        population[:] = offspring
        convergence[generation] = population[0].fitness
    return population, convergence


class DeIndividual(BaseIndividual):
    def __init__(self, dim, bounds=(-30, 30)):
        super(DeIndividual, self).__init__(dim, bounds)
        self.variance = None

    def mutate(self, mutation_p=0.8, a=None, b=None):
        if a is None or b is None:
            raise Exception("Requires three mates")

        offspring = copy.deepcopy(self)

        offspring.x = offspring.x + 0.2 * np.subtract(np.array(a.x), np.array(b.x))
        return offspring

    def crossover(self, other, crossover_p=0.2):
        n = len(self.x)
        j = random.randint(0, n)
        c = copy.deepcopy(self)
        c.x = np.asarray([self.x[i] if r <= crossover_p or i == j else other.x[i]
                          for i, r in enumerate(np.random.uniform(size=n, low=0, high=1))])
        return c

    def select(self, other):
        if self.fitness < other.fitness:
            return self
        else:
            return other
