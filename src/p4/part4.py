import os.path
import time
import warnings
import pygraphviz

from sklearn.exceptions import ConvergenceWarning
from Configuration.IDGP_main import *
from Configuration.tools import configure_gp
from src.common import utils
from src.common.utils import print_header

warnings.filterwarnings(action='ignore', category=ConvergenceWarning, module='sklearn')


def load_npy():
    path = utils.get_common_path("data", "FEI-dataset")
    X_train = np.load(os.path.join(path, dataset, dataset + '_train_data.npy')) / 255.0
    y_train = np.load(os.path.join(path, dataset, dataset + '_train_label.npy'))
    X_test = np.load(os.path.join(path, dataset, dataset + '_test_data.npy')) / 255.0
    y_test = np.load(os.path.join(path, dataset, dataset + '_test_label.npy'))
    return X_train, y_train, X_test, y_test


def set_features(X, y):
    return np.concatenate(
        (toolbox.scale_fit(
            individual=hof[0],
            x=X,
            y=y),
         y.reshape((-1, 1))), axis=1)


if __name__ == "__main__":
    for dataset in ['f1', 'f2']:
        X_train, y_train, X_test, y_test = load_npy()

        t0 = time.process_time()

        # conf = configure_gp() # --- Moved to IDGP_main.py
        toolbox = make_toolbox(dataset, X_train, y_train)
        pop, log, hof = GPMain(0, toolbox)

        t1 = time.process_time()
        delta0 = t1 - t0

        print(scale_fit(individual=hof[0], x=X_test, y=y_test, toolbox=toolbox))
        print(y_test.reshape((-1, 1)).shape)

        test_sample = set_features(X_test, y_test)
        train_sample = set_features(X_train, y_train)

        results = eval(individual=hof[0],
                       x=X_train,
                       y=y_train,
                       toolbox=toolbox,
                       cross_validate=False,
                       x_test=X_test,
                       y_test=y_test)

        utils.export(train_sample,
                     text=os.path.basename(__file__),
                     name=dataset + '_train_features')

        utils.export(test_sample,
                     text=os.path.basename(__file__),
                     name=dataset + '_test_features')
        t2 = time.process_time()
        delta1 = t2 - t1

        print_header(f"Dataset: {dataset} Results")
        print('Best individual ', hof[0])
        print('Test results  ', results)
        # Print rounded time taken in seconds
        print("Training took", round(delta0, 2), "seconds. Testing took", round(delta1, 2), "seconds.")

        G = pygraphviz.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'

        # Add a single square node with the meta data
        G.add_node(0)
        G.get_node(0).attr["label"] = "part 4" + dataset + " Graph"
        G.get_node(0).attr["fillcolor"] = "lightblue"

        utils.render_graph([hof], G=G, part="part4", seed=dataset, toolbox=toolbox)
