import operator
import random

import numpy as np
from numpy import floor, log10


def round(x, sf):
    """
    round(x, sf)

    Round a number to a specified number of significant figures.

    Parameters
    ----------
    x : float
        The number to be rounded.
    sf : int
        The number of significant figures to round to.

    Returns
    -------
    float
        The rounded number.

    Examples
    --------
    >>> round(0.0012345, 3)
    0.00123
    """
    return np.round(x, sf - int(floor(log10(abs(x)))) - 1)


def fitness(ind, f, bounds=(-30, 30)):
    """
    # Check if any of the values in the individual are outside the bounds
    # If they are, return infinity
    # If they are not, return the fitness value
    @param ind: Target to check
    @param f: Fitness function
    @param bounds: Bounds of the problem
    @return: Fitness value
    """
    if any([x > np.max(bounds) or x < np.min(bounds) for x in ind.x]):
        return np.inf
    return f(ind.x)


def tournament_selection(all, N, q):
    """

    Select population_size individuals from a list of individuals using tournament selection.

    Parameters
    ----------
    all : list
        The list of individuals to select from.
    N : int
        The number of individuals to select.
    q : int
        The number of individuals to select from.

    Returns
    -------
    list
        The list of selected individuals.

    """
    tmp = all.copy()
    all.sort(key=lambda ind: ind.selection(list(set(tmp[:]) - {ind}), N * q), reverse=True)
    return all[:N]


def select(population, N):
    """
        select(population, population_size)

        Select the fittest population_size individuals from a population.

        Parameters
        ----------
        population : list
            The population to select from.
        N : int
            The number of individuals to select.

        Returns
        -------
        list
            The fittest population_size individuals.


        """
    population.sort(key=operator.attrgetter('fitness'))
    return population[:N]


def round_metrics(fitness_values):
    """
        round_metrics(fitness_values)

        Round the metrics of a list of fitness values.

        Parameters
        ----------
        fitness_values : list
            A list of fitness values.

        Returns
        -------
        dict
            A dictionary containing the rounded metrics of the fitness values.

        Examples
        --------
        >>> round_metrics([0.0042345, 0.0012345, 0.0012345, 0.013])
        {'mean': 0.005, 'min': 0.001, 'max': 0.013, 'std': 0.005}
        """
    metrics = {
        "mean": np.round(np.mean(fitness_values), 3),
        "min": np.round(np.min(fitness_values), 3),
        "max": np.round(np.max(fitness_values), 3),
        "std": np.round(np.std(fitness_values), 3),
    }
    return metrics


class BaseIndividual:
    """
        BaseIndividual(dim, bounds=(-30, 30))

        Base class for an individual in the population.

        Parameters
        ----------
        dim : int
            The dimension of the individual.
        bounds : tuple
            The lower and upper bounds of the individual.

        Attributes
        ----------
        D : int
            The dimension of the individual.
        x : list
            The position of the individual.
        r : numpy.ndarray
            The step size of the individual.
        fitness : float
            The fitness of the individual.
        tau : float
            The tau parameter of the individual.
        tau_prime : float
            The tau prime parameter of the individual.

        Methods
        -------
        selection(tournament_pool, q)
            Select the individual from a tournament pool.
        mutate(mu, sigma, indpb)
            Mutate the individual.
        """

    def __init__(self, dim, bounds=(-30, 30)):
        self.D = dim
        self.x = [random.uniform(np.min(bounds), np.max(bounds)) for _ in range(dim)]
        self.r = np.array([3.0 for _ in range(dim)])
        self.fitness = np.inf
        self.tau = 1 / np.sqrt(2 * np.sqrt(dim))
        self.tau_prime = 1 / np.sqrt(2 * dim)

    def selection(self, tournament_pool, q, ):
        opponents = random.sample(tournament_pool, int(q))
        opponents += [self]
        opponents.sort(key=operator.attrgetter('fitness'))
        self_index = opponents.index(self)

        return len(opponents) - self_index

    # Mutate with Cauchy dist
    def mutate(self, mu, sigma, indpb):
        self.x += self.r * np.random.standard_cauchy(size=self.D)
        self.r *= np.exp(self.tau_prime * np.random.normal(0, 1) +
                         self.tau * np.random.standard_normal(size=self.D))

        return self

    def __repr__(self):
        return f"Individual(x={self.x}, function={self.fitness})"

    def __str__(self):
        return f"Individual(x={self.x}, function={self.fitness})"
