# Adaptation of Research to run cooperative co-evolution genetic programming on the symbolic regression problem:
#  function (x) = { 1/x + sin x , x > 0
#            2/x + x^2 + 3.0 , x ≤ 0
#
# In project 1, we assume that there is no prior knowledge about the target model.
# In this project, the assumption is changed.
#
# Instead of knowing nothing, we know that the target model is a piecewise function, with two sub-functions:
#   f1(x) for x > 0 and f2(x) for x ≤ 0.
# In other words, we know that the target function is:
#
# function (x) = { f1(x), x > 0
#           f2(x), x ≤ 0 (5)
#
# This question is to develop a Cooperative Co-evolution GP (CCGP) to solve this symbolic
# regression problem.
#
# The CCGP should contain two sub-populations, one for f1(x) and the other for f2(x).
# You can use a GP library.
#
# You should:
# • Determine and describe the terminal set and the function set of each sub-population.
# • Design the fitness function and the fitness evaluation method for each sub-population.
# • Set the necessary parameters, such as:
#                       sub-population size,
#                       maximum tree depth,
#                       termination criteria,
#                       crossover and
#                       mutation rates.
# • Run the implemented CCGP for 5 times with different random seeds.
#
# Report the best genetic programs (their structure and performance) of each of the 5 runs.
# Present your observations and discussions and draw your conclusion
import random
import math
import operator

import numpy
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap.algorithms import varOr

from src.common.utils import print_header, render_graph
from Configuration.tools import configure_symbreg, make_primitive_set

PARAMS = configure_symbreg()
PSET = make_primitive_set()


def evaluate(individual, points):
    actual = toolbox.compile(expr=individual)

    def target(x):
        """
        The target function we are trying to approximate
        """
        if x > 0:
            return 1 / x + math.sin(x)
        return 2 * x + x ** 2 + 3.0

    RSS = math.fsum([(actual(x) - target(x)) ** 2 for x in points])
    # residual sum of squares
    return RSS / len(points),


def make_toolbox():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    # Generate individual using half (full) and half (grow) method
    toolbox.register("expr", gp.genHalfAndHalf, pset=PSET, min_=PARAMS["init_min_depth"], max_=PARAMS["init_max_depth"])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    # Generate population of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=PSET)

    # Evaluate over x = [-10, 15] step by 1/5
    toolbox.register("evaluate", evaluate, points=[x / 5. for x in range(-50, 75)])
    toolbox.register("evaluatePos", evaluate, points=[x / 5. for x in range(1, 75)])
    toolbox.register("evaluateNeg", evaluate, points=[x / 5. for x in range(-50, 1)])
    # Tournament selection, 3 participants
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    # Mutate new subtrees
    toolbox.register("expr_mut", gp.genFull, min_=PARAMS["mutate_min_depth"], max_=PARAMS["mutate_max_depth"])
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=PSET)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=PARAMS["max_depth"]))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=PARAMS["max_depth"]))

    return toolbox


toolbox = make_toolbox()


def modEaMuPlusLambda(seed=None):
    """
    # Modified version of DEAP's eaMuPlusLambda, see
    # eaMuPlusLambda = algorithms.eaMuPlusLambda
    @param seed: Random Seed to use
    @return: -
    """
    evals, hofs, pops, stats = initialise(seed)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields

    for i1, (pop1, hof1, evaluatePos) in enumerate(zip(pops, hofs, evals)):
        # Evaluate the individuals with an invalid fitness
        ind1 = [ind2 for ind2 in pop1 if not ind2.fitness.valid]
        fitnesses1 = toolbox.map(evaluatePos, ind1)
        for ind2, fit1 in zip(ind1, fitnesses1):
            ind2.fitness.values = fit1

        # Update the hall of fame with the generated individuals
        hof1.update(pop1)

    # Begin the generational process
    for epoch in range(0, PARAMS["epochs"]):
        for i, (pop, hof, evaluate) in enumerate(zip(pops, hofs, evals)):
            # Vary the population
            offspring = varOr(pop, toolbox, PARAMS["mu"], PARAMS["p_cross"], PARAMS["p_mutate"])

            # Evaluate the individuals with an invalidated fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            hof.update(offspring)

            # Keep best and fill remainder with offspring
            remaining = PARAMS["mu"] - PARAMS["n_elite"]
            pop[:] = tools.selBest(pop, PARAMS["n_elite"]) + tools.selBest(offspring, remaining)

            # Update the statistics with the new pop
            record = stats.compile(pop)
            logbook.record(gen=epoch, pop=i, nevals=len(invalid_ind), **record)

            if PARAMS["verbose"]:
                print(logbook.stream)

    render_graph(hofs, seed, "part3", toolbox=toolbox, PARAMS=PARAMS)


def initialise(seed):
    random.seed(seed)
    np.random.seed(seed)
    pops = [toolbox.population(n=PARAMS["mu"]), toolbox.population(n=PARAMS["mu"])]
    hofs = [tools.HallOfFame(1), tools.HallOfFame(1)]
    evals = [toolbox.evaluatePos, toolbox.evaluateNeg]
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    return evals, hofs, pops, stats


if __name__ == '__main__':

    for seed in [6, 42, 512, 1337, 2048]:
        print_header("Seed: " + str(seed))
        modEaMuPlusLambda(seed)
