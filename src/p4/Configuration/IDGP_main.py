# python packages
import operator
import random

import IDGP.evalGP_main as evalGP
import IDGP.feature_function as fe_fs
# only for strongly typed GP
import IDGP.gp_restrict as gp_restrict
import numpy as np

from Configuration.tools import configure_gp
from IDGP.strongGPDataType import Int1, Int2, Int3, Img, Region, Vector
# deap package
from deap import base, creator, tools, gp
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

# parameters:


conf = configure_gp()

population = conf['population']
generation = conf['generation']
cxProb = conf['cxProb']
mutProb = conf['mutProb']
elitismProb = conf['elitismProb']
totalRuns = conf['totalRuns']
initialMinDepth = conf['initialMinDepth']
initialMaxDepth = conf['initialMaxDepth']
maxDepth = conf['maxDepth']
x = conf['x']
y = conf['y']


def make_primitive_set(data, x):
    bound1, bound2 = x[0, :, :].shape
    pset = gp.PrimitiveSetTyped('MAIN', [Img], Vector, prefix='Image')

    # Feature concatenation
    pset.addPrimitive(fe_fs.root_con, [Vector, Vector], Vector, name='FeaCon2')
    pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector], Vector, name='FeaCon3')

    # Global feature extraction
    pset.addPrimitive(fe_fs.all_dif, [Img], Vector, name='Global_DIF')
    pset.addPrimitive(fe_fs.all_histogram, [Img], Vector, name='Global_Histogram')
    pset.addPrimitive(fe_fs.global_hog, [Img], Vector, name='Global_HOG')
    pset.addPrimitive(fe_fs.all_lbp, [Img], Vector, name='Global_uLBP')
    pset.addPrimitive(fe_fs.all_sift, [Img], Vector, name='Global_SIFT')

    # Local feature extraction
    pset.addPrimitive(fe_fs.all_dif, [Region], Vector, name='Local_DIF')
    pset.addPrimitive(fe_fs.all_histogram, [Region], Vector, name='Local_Histogram')
    pset.addPrimitive(fe_fs.local_hog, [Region], Vector, name='Local_HOG')
    pset.addPrimitive(fe_fs.all_lbp, [Region], Vector, name='Local_uLBP')
    pset.addPrimitive(fe_fs.all_sift, [Region], Vector, name='Local_SIFT')

    # Region detection operators
    pset.addPrimitive(fe_fs.regionS, [Img, Int1, Int2, Int3], Region, name='Region_S')
    pset.addPrimitive(fe_fs.regionR, [Img, Int1, Int2, Int3, Int3], Region, name='Region_R')

    # Terminals
    pset.renameArguments(ARG0='Grey')

    random_string = ''.join(random.choice('0123456789ABCDEF') for i in range(8))
    # check if the terminal is already added
    pset = addEphemerals(pset, random_string, bound1, bound2)
    # primitives.addEphemeralConstant('X'+random_string, lambda: random.randint(0, bound1 - 20), Int1)
    # primitives.addEphemeralConstant('Y'+random_string, lambda: random.randint(0, bound2 - 20), Int2)
    # primitives.addEphemeralConstant('Size'+random_string, lambda: random.randint(20, 51), Int3)
    return pset


def addEphemerals(primitives, random_string, bound1, bound2):
    try:
        primitives.addEphemeralConstant('X' + random_string, lambda: random.randint(0, bound1 - 20), Int1)
        primitives.addEphemeralConstant('Y' + random_string, lambda: random.randint(0, bound2 - 20), Int2)
        primitives.addEphemeralConstant('Size' + random_string, lambda: random.randint(20, 51), Int3)
    except:
        r = ''.join(random.choice('0123456789ABCDEF') for i in range(8))
        primitives = addEphemerals(primitives, r, bound1, bound2)
    return primitives


# fitness eval
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()


def make_toolbox(df, x, y, **kwargs):
    primitives = make_primitive_set(df, x)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=primitives, min_=initialMinDepth, max_=initialMaxDepth)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=primitives)
    toolbox.register("mapp", map)
    toolbox.register("classifier", LinearSVC, max_iter=100)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("selectElitism", tools.selBest)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitives)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
    toolbox.register("transform", scale_fit, toolbox=toolbox)
    toolbox.register("evaluate", eval, toolbox=toolbox, x=x, y=y)
    return toolbox


def eval(individual, toolbox, x, y, y_test=None, x_test=None, cross_validate=True):
    n1 = toolbox.scale_fit(individual, x=x, y=y)
    LSVM = toolbox.classifier()
    if cross_validate:
        score = cross_val_score(LSVM, n1, y, scoring="f1", cv=3).mean()
        metrics = round(score, 8),  # TUPLE
    else:
        n2 = toolbox.scale_fit(individual, x=x_test, y=y_test)
        LSVM.fit(n1, y)
        pred = LSVM.predict(n2)
        m1 = f1_score(y_test, pred)
        m2 = LSVM.score(n2, y_test)
        metrics = (m1, m2)
    return metrics


def GPMain(randomSeeds, toolbox):
    random.seed(randomSeeds)

    pop = toolbox.population(population)
    hof = tools.HallOfFame(10)
    log = tools.Logbook()
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    log.header = ["gen", "evals"] + stats.fields

    pop, log = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, elitismProb, generation,
                               stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof


def scale_fit(individual, toolbox, x, y):
    """
    Scale the features
    @param individual:
    @param toolbox:
    @param x:
    @param y:
    @return:
    """
    f = toolbox.compile(expr=individual)
    d = []
    for i in range(0, len(y)):
        d.append(np.asarray(f(x[i, :, :])))
    d = np.asarray(d)
    scaler = preprocessing.MinMaxScaler()
    scaled = scaler.fit_transform(np.asarray(d))

    return scaled
