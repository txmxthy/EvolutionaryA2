import os
from pprint import pprint

import pandas as pd
import pygraphviz as pgv
from deap import gp


def print_header(string, sep="="):
    """
    Print a header with a string in the middle
    """
    print(sep * 40)
    print(string.center(40))
    print(sep * 40)


def get_common_path(dir, name):
    """
    Get the path to the common resource
    """
    path = os.path.join(os.path.dirname(__file__), dir, name)
    return path


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def render(plot, text, name):
    # If there is an extension on the text, remove it
    if "." in text:
        text = text.split(".")[0]

    path = get_common_path("output", text)
    create_dir(path)
    plot.savefig(os.path.join(path, name) + ".jpg", bbox_inches='tight')


def export(features, text, name):
    if "." in text:
        text = text.split(".")[0]
    path = get_common_path("output", text)
    create_dir(path)
    pd.DataFrame(features).to_csv(os.path.join(path, f"{name}.csv"), header=None, index=None)


def load_knapsack():
    instances = get_common_path("data", "knapsack-data")
    instances = [instances + "/" + file for file in os.listdir(instances)]
    Loaded = {}
    for instance in instances:
        name = instance.split("/")[-1]
        Loaded[name] = {}
        problem = pd.read_csv(instance, sep=" ", header=None)
        Loaded[name]["Count"] = problem.iloc[0, 0]
        Loaded[name]["Limit"] = problem.iloc[0, 1]
        Loaded[name]["Items"] = pd.DataFrame(problem.iloc[1:, :].values, columns=["value", "weight"])
    return Loaded


def render_graph(hofs, seed, part, G=None, toolbox=None, PARAMS=None):
    """
    Render the final expression as a graph and save it to a file
    @param hofs : list
        A list of hall of fame individuals.
    @param seed : int
        The seed value used to generate the hall of fame individuals.

    """

    if G is None:  # Default graph for if statement // Hacky didnt think I would reuse this for part 4
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        G.add_node(0)
        G.get_node(0).attr["label"] = "if"
        G.get_node(0).attr["fillcolor"] = "#cccccc"

        # LEFT PARENT
        G.add_node(1)
        G.get_node(1).attr["label"] = "x > 0"
        G.get_node(1).attr["fillcolor"] = "#6a7ea6"
        G.add_edge(0, 1)

        # RIGHT PARENT
        G.add_node(2)
        G.get_node(2).attr["label"] = "x ≤ 0"
        G.get_node(2).attr["fillcolor"] = "#7654ba"
        G.add_edge(0, 2)

    colors = ["#6a7ea6", "#7654ba"]

    print_header("Results, seed = {}".format(seed))
    for pop, hof in enumerate(hofs):
        best_program = toolbox.compile(expr=hof.items[0])

        print("Population {} best fitness: {} size {}.".format(pop + 1, hof.keys[0], len(hof.items[0])))

        if PARAMS is not None:
            if PARAMS["verbose"]:
                print(str(hof.items[0]))
                pprint([best_program(x / 5.) for x in range(-25, 75)])

        nodes, edges, labels = gp.graph(hof.items[0])
        base = G.number_of_nodes()
        G.add_nodes_from([base + i for i in nodes])
        G.add_edges_from([(base + i, base + j) for (i, j) in edges])
        print(labels)
        # Add labels to nodes and split into a
        for i in nodes:
            G.get_node(base + i).attr["label"] = labels[i]
            G.get_node(base + i).attr["fillcolor"] = colors[pop]
            # I wrote this but it doesnt work between my machines
            # Removed Shape for now, cant be bothered fully implementing as it is unstable
            # # Shape
            # # if is float Absolute hack job
            # if labels[i] is not None and labels[i].find(".") != -1:
            #     G.get_node(base + i).attr["shape"] = "box"
            # else:
            #     G.get_node(base + i).attr["shape"] = "ellipse"

        # Find nodes with no incoming edges
        if len(hofs) > 1:
            roots = []
            for n in nodes:
                if not any([e[1] == n for e in edges]):
                    roots.append(n)
            G.add_edge(pop + 1, roots[0] + base)

    G.layout(prog="dot")

    path = get_common_path("output", part + "/graph_{}.png".format(seed))

    # Make sure output directory exists, create if not
    os.makedirs(os.path.dirname(path), exist_ok=True)
    G.draw(path)
