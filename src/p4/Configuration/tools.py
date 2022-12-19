def configure_gp():
    """
    Configure the GP Parameters
    @return:
    """

    # Defaults
    population = 44
    generation = 88
    cxProb = 0.8
    mutProb = 0.1
    elitismProb = 0.1
    totalRuns = 1
    initialMinDepth = 1
    initialMaxDepth = 4
    maxDepth = 10
    x = None
    y = None

    # Get user input
    # Get the parameters from user input, if enter is pressed, use the default value
    # if user wants to continue with default values, just press enter
    custom = input("Use custom parameters? (y/n): ")
    if custom == "y":
        population = int(input("Population size: "))
        generation = int(input("Number of generations: "))
        cxProb = float(input("Crossover probability: "))
        mutProb = float(input("Mutation probability: "))
        elitismProb = float(input("Elitism probability: "))
        totalRuns = int(input("Number of runs: "))
        initialMinDepth = int(input("Minimum initial depth: "))
        initialMaxDepth = int(input("Maximum initial depth: "))
        maxDepth = int(input("Maximum depth: "))

    config = {
        'population': population,
        'generation': generation,
        'cxProb': cxProb,
        'mutProb': mutProb,
        'elitismProb': elitismProb,
        'totalRuns': totalRuns,
        'initialMinDepth': initialMinDepth,
        'initialMaxDepth': initialMaxDepth,
        'maxDepth': maxDepth,
        'x': x,
        'y': y
    }

    return config
