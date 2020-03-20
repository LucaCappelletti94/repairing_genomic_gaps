from repairing_genomic_gaps import cae_1000, cae_500, cae_200
from repairing_genomic_gaps import build_multivariate_dataset_cae, build_synthetic_dataset_cae
from repairing_genomic_gaps import train_model
import os
from notipy_me import Notipy

models = [
    (cae_1000, 1000),
    (cae_500, 500),
    (cae_200, 200),
]

weights = [
    10,
    2,
]


datasets = [
    (build_multivariate_dataset_cae, "multivariate_gaps"),
    (build_synthetic_dataset_cae, "single_gap"),
]

for dataset, dataset_name in datasets:
    for weight in weights:
        path = "{}_with_weight_{}".format(dataset_name, weight)
        if os.path.exists(path):
            continue
        for build_model, window_size in models:
            with Notipy(
                task_name="Model {model_name} trained on {dataset_name} with max weight {weight}".format(
                    model_name=build_model.__name__,
                    dataset_name=dataset_name,
                    weight=weight,
                )
            ):

                model = build_model(
                    use_weighted=True,
                    _max = weight,
                )
                train, test = dataset(window_size)
                model = train_model(model, train, test, path=path)