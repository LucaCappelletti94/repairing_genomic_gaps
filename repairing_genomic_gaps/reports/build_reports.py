from ..models import cae_200, cae_500, cae_1000, cnn_200, cnn_500, cnn_1000
from ..utils import get_model_history_path, get_model_weights_path
from ..datasets import build_multivariate_dataset_cae, build_synthetic_dataset_cae
from ..datasets import build_multivariate_dataset_cnn, build_synthetic_dataset_cnn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
from typing import Dict, List, Callable
from tensorflow.keras.utils import Sequence
from tensorflow.keras import Model
from .report_utils import cae_report, cnn_report, flat_report
import warnings

warnings.simplefilter("ignore")

models = {
    "cae": {
        200: cae_200,
        500: cae_500,
        1000: cae_1000,
    },
    "cnn": {
        200: cnn_200,
        500: cnn_500,
        1000: cnn_1000
    }
}

datasets = {
    "cae": [
        build_synthetic_dataset_cae,
        build_multivariate_dataset_cae,
    ],
    "cnn": [
        build_synthetic_dataset_cnn,
        build_multivariate_dataset_cnn,
    ]
}

report_types = {
    "cae": cae_report,
    "cnn": cnn_report
}

def get_report_path(model, dataset, run_type):
    return "report_{model}_{dataset}_{run_type}.csv".format(
        model=model.name,
        dataset=dataset.__name__,
        run_type=run_type
    )

def execute_report(report, model, dataset, run_type, sequence):
    path = get_report_path(model, dataset, run_type)

    if os.path.exists(path):
        return

    pd.DataFrame(flat_report(
        build_report(model, report, sequence),
        model,
        dataset,
        run_type
    )).to_csv(path)


def build_report(model: Model, report: Callable, sequence: Sequence):
    sequence.on_epoch_end()
    X, y = np.vstack([
        sequence[batch]
        for batch in tqdm(range(min(100, sequence.steps_per_epoch)), desc="Rendering batches", leave=False)
    ])
    return report(y, model.predict(X))


def build_reports(**dataset_kwargs):
    for model_type in tqdm(models, desc="Model types", leave=False):
        report = report_types[model_type]
        for window_size, build_model in tqdm(models[model_type].items(), desc="Models", leave=False):
            single_gap_dataset, multivariate_dataset = datasets[model_type]
            single_train, single_test = single_gap_dataset(window_size, **dataset_kwargs)
            multivariate_train, multivariate_test = multivariate_dataset(window_size, **dataset_kwargs)
            #bio = biological(window_size)
            model = build_model(verbose=False)
            model.load_weights(get_model_weights_path(model, path="single_gap"))

            execute_report(
                report, model, single_gap_dataset, "single gap test", single_test
            )

            execute_report(
                report, model, single_gap_dataset, "single gap train", single_train
            )

            execute_report(
                report, model, multivariate_dataset, "multivariate gaps test", multivariate_test
            )

            execute_report(
                report, model, multivariate_dataset, "multivariate gaps train", multivariate_train
            )
            
            
            # reports += flat_report(
            ###################################
            # TODO: UPDATE THE DATASET AS SOON
            # AS IT BECOMES AVAILABLE!!!
            ###################################
            #build_report(model, report, valid),
            # model,
            # biological,
            #"biological validation (hg19/38)"
            # )
