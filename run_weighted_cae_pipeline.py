from repairing_genomic_gaps import cae_1000, cae_500, cae_200
from repairing_genomic_gaps import build_multivariate_dataset_cae, build_synthetic_dataset_cae
from repairing_genomic_gaps import train_model

import multiprocessing as mp
from notipy_me import Notipy
from itertools import product

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class mp.pool.Pool instead of mp.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(mp.pool.Pool):
    Process = NoDaemonProcess

models = [
    (cae_1000, 1000),
    (cae_500, 500),
    (cae_200, 200),
]

weights = [
    10,
    #2,
]


def executor(build_model, window_size, weight):
    datasets = [
        (build_multivariate_dataset_cae, "multivariate_gaps"),
        (build_synthetic_dataset_cae, "single_gap"),
    ]
    for dataset, dataset_name in datasets:
        with Notipy(
            task_name="Model {model_name} trained on {dataset_name} with max weight {weight}".format(
                model_name=build_model.__name__,
                dataset_name=dataset_name,
                weight=weight,
            )
        ):
            # model = build_model(
            #     use_weighted=True,
            #     _max = weight,
            # )
            train, test = dataset(window_size)
            #path = "{}_with_weight_{}".format(dataset_name, weight)
            #model = train_model(model, train, test, path=path)

tasks = [
    (*model, weight)
    for model, weight in product(models, weights)
]

with MyPool(2) as p:
    p.starmap(executor, tasks)
