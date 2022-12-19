"""
This code is the taken from the examples of the DEAP library and the deap documentation.
See: https://deap.readthedocs.io/en/master/examples/coev_coop.html
This is the condensed version of the code, and what the actual submission for p3 is based on.

The full version can be found online at:
https://github.com/DEAP/deap/blob/fa3abe189e71cb7afef71e0acaaf1a635f083281/examples/coev/symbreg.py
"""

import random
import sys

import numpy
import operator
import math
import random

import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import gp

import src.p1.Algorithms.base

sys.path.append("")

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("IndGA", list, fitness=creator.FitnessMax)

toolbox_ga = base.Toolbox()
toolbox_ga.register("float", random.uniform, -1, 1)
toolbox_ga.register("individual", tools.initRepeat, creator.IndGA, toolbox_ga.float, 10)
toolbox_ga.register("population", tools.initRepeat, list, toolbox_ga.individual)
toolbox_ga.register("select", tools.selTournament, tournsize=3)
toolbox_ga.register("mate", tools.cxTwoPoint)
toolbox_ga.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.05)


# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox_gp = base.Toolbox()
toolbox_gp.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox_gp.register("individual", tools.initIterate, creator.Individual, toolbox_gp.expr)
toolbox_gp.register("population", tools.initRepeat, list, toolbox_gp.individual)
toolbox_gp.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox_gp.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = []
    for x in points:
        sqerrors.append((func(x) - f(x)) ** 2)
    return math.fsum(sqerrors) / len(points),


def f(x):
    if x > 0:
        return 1 / x + np.sin(x)
    else:
        return 2 / x + x ** 2 + 3.0


toolbox_gp.register("evaluate", evalSymbReg, points=[x / 10. for x in range(-10, 10)])
toolbox_gp.register("select", tools.selTournament, tournsize=3)
toolbox_gp.register("mate", gp.cxOnePoint)
toolbox_gp.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox_gp.register("mutate", gp.mutUniform, expr=toolbox_gp.expr_mut, pset=pset)

toolbox_gp.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox_gp.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def main():
    pop_ga = toolbox_ga.population(n=200)
    pop_gp = toolbox_gp.population(n=200)

    stats = tools.Statistics(lambda ind: src.p1.Algorithms.base.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "type", "evals", "std", "min", "avg", "max"

    best_ga = tools.selRandom(pop_ga, 1)[0]
    best_gp = tools.selRandom(pop_gp, 1)[0]

    for ind in pop_gp:
        src.p1.Algorithms.base.fitness.values = toolbox_gp.evaluate(ind, points=best_ga)

    for ind in pop_ga:
        src.p1.Algorithms.base.fitness.values = toolbox_gp.evaluate(best_gp, points=ind)

    record = stats.compile(pop_ga)
    logbook.record(gen=0, type='ga', evals=len(pop_ga), **record)

    record = stats.compile(pop_gp)
    logbook.record(gen=0, type='gp', evals=len(pop_gp), **record)

    print(logbook.stream)

    CXPB, MUTPB, NGEN = 0.5, 0.2, 50

    # Begin the evolution
    best_ga, best_gp, pop_ga, pop_gp = do_evolution(CXPB, MUTPB, NGEN, best_ga, best_gp, logbook, pop_ga, pop_gp, stats)

    print("Best individual GA is %s, %s" % (best_ga, src.p1.Algorithms.base.fitness.values))
    print("Best individual GP is %s, %s" % (best_gp, src.p1.Algorithms.base.fitness.values))

    return pop_ga, pop_gp, best_ga, best_gp, logbook


def do_evolution(CXPB, MUTPB, NGEN, best_ga, best_gp, logbook, pop_ga, pop_gp, stats):
    for g in range(1, NGEN):
        # Select and clone the offspring
        off_ga = toolbox_ga.select(pop_ga, len(pop_ga))
        off_gp = toolbox_gp.select(pop_gp, len(pop_gp))
        off_ga = [toolbox_ga.clone(ind) for ind in off_ga]
        off_gp = [toolbox_gp.clone(ind) for ind in off_gp]

        # Apply crossover and mutation
        for ind1, ind2 in zip(off_ga[::2], off_ga[1::2]):
            if random.random() < CXPB:
                toolbox_ga.mate(ind1, ind2)
                del src.p1.Algorithms.base.fitness.values
                del src.p1.Algorithms.base.fitness.values

        for ind1, ind2 in zip(off_gp[::2], off_gp[1::2]):
            if random.random() < CXPB:
                toolbox_gp.mate(ind1, ind2)
                del src.p1.Algorithms.base.fitness.values
                del src.p1.Algorithms.base.fitness.values

        for ind in off_ga:
            if random.random() < MUTPB:
                toolbox_ga.mutate(ind)
                del src.p1.Algorithms.base.fitness.values

        for ind in off_gp:
            if random.random() < MUTPB:
                toolbox_gp.mutate(ind)
                del src.p1.Algorithms.base.fitness.values

        # Evaluate the individuals with an invalid fitness
        for ind in off_ga:
            src.p1.Algorithms.base.fitness.values = toolbox_gp.evaluate(best_gp, points=ind)

        for ind in off_gp:
            src.p1.Algorithms.base.fitness.values = toolbox_gp.evaluate(ind, points=best_ga)

        # Replace the old population by the offspring
        pop_ga = off_ga
        pop_gp = off_gp

        record = stats.compile(pop_ga)
        logbook.record(gen=g, type='ga', evals=len(pop_ga), **record)

        record = stats.compile(pop_gp)
        logbook.record(gen=g, type='gp', evals=len(pop_gp), **record)
        print(logbook.stream)

        best_ga = tools.selBest(pop_ga, 1)[0]
        best_gp = tools.selBest(pop_gp, 1)[0]
    return best_ga, best_gp, pop_ga, pop_gp


if __name__ == "__main__":
    main()
