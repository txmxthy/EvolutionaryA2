import math
import operator
import random

from deap import gp


def configure_symbreg():
    """
    verbose: Boolean, whether to print out the progress of the algorithm.
    mu: Integer, the number of individuals in the population.
    p_cross: Float, the probability of crossover.
    p_mutate: Float, the probability of mutation.
    n_elite: Integer, the number of individuals to select for the next generation.
    epochs: Integer, the number of epochs to run the algorithm for.
    init_min_depth: Integer, the minimum depth of the initial population.
    init_max_depth: Integer, the maximum depth of the initial population.
    max_depth: Integer, the maximum depth of the trees.
    mutate_min_depth: Integer, the minimum depth of the trees after mutation.
    mutate_max_depth: Integer, the maximum depth of the trees after mutation.
    """

    # Defaults
    verbose = False
    mu = 1000
    p_cross = 0.8
    p_mutate = 0.2
    n_elite = int(mu * 0.1)
    epochs = 50
    init_min_depth = 1
    init_max_depth = 3
    max_depth = 7
    mutate_min_depth = 1
    mutate_max_depth = 3

    # Get the parameters from user input, if enter is pressed, use the default value
    # if user wants to continue with default values, just press enter
    custom = input("Use custom parameters? (y/n): ")
    if custom == "y":
        # Convert verbose to boolean
        verbose = input("Verbose? (y/n): ") == "y"

        mu = int(input(
            "Population size (mu - default {}): ".format(mu)) or mu)
        p_cross = float(input(
            "Crossover probability (p_cross - default {}): ".format(p_cross)) or p_cross)
        p_mutate = float(input(
            "Mutation probability (p_mutate - default {}): ".format(p_mutate)) or p_mutate)
        n_elite = int(input(
            "Number of elite individuals (n_elite - default {}): ".format(n_elite)) or n_elite)
        epochs = int(input(
            "Number of epochs (epochs - default {}): ".format(epochs)) or epochs)
        init_min_depth = int(input(
            "Initial minimum depth (init_min_depth - default {}): ".format(init_min_depth)) or init_min_depth)
        init_max_depth = int(input(
            "Initial maximum depth (init_max_depth - default {}): ".format(init_max_depth)) or init_max_depth)
        max_depth = int(input(
            "Maximum depth (max_depth - default {}): ".format(max_depth)) or max_depth)
        mutate_min_depth = int(input(
            "Mutation minimum depth (mutate_min_depth - default {}): ".format(mutate_min_depth)) or mutate_min_depth)
        mutate_max_depth = int(input(
            "Mutation maximum depth (mutate_max_depth - default {}): ".format(mutate_max_depth)) or mutate_max_depth)
    else:
        print("Using default parameters")

    print("Verbose: {}".format(verbose))
    # Set the parameters as global variables in a dict

    PARAMS = {
        "verbose": verbose,
        "mu": mu,
        "p_cross": p_cross,
        "p_mutate": p_mutate,
        "n_elite": n_elite,
        "epochs": epochs,
        "init_min_depth": init_min_depth,
        "init_max_depth": init_max_depth,
        "max_depth": max_depth,
        "mutate_min_depth": mutate_min_depth,
        "mutate_max_depth": mutate_max_depth
    }

    return PARAMS


def make_primitive_set():
    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1

    def protectedPow(left, right):
        try:
            return left ** right
        except OverflowError:
            return 1

    def protectedSquare(x):
        try:
            return math.pow(x, 2)
        except OverflowError:
            return float('inf')

    pset = gp.PrimitiveSetTyped("main", [float], float, "x")
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(protectedDiv, [float, float], float)
    pset.addPrimitive(math.cos, [float], float)
    # pset.addPrimitive(math.sin, [float], float)
    # pset.addPrimitive(math.tan, [float], float)
    # pset.addPrimitive(math.log, [float], float)
    # pset.addPrimitive(math.exp, [float], float)
    # pset.addPrimitive(math.sqrt, [float], float)
    # pset.addPrimitive(protectedPow, [float, float], float)
    pset.addPrimitive(protectedSquare, [float], float)
    # pset.addPrimitive(math.fabs, [float], float)
    pset.addEphemeralConstant("rand1", lambda: random.random() * 100, float)

    return pset
