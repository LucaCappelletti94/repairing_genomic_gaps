from ..models import cae_200, cae_500, cae_1000, cnn_200, cnn_500, cnn_1000
from ..utils import get_model_history_path, get_model_weights_path
from ..datasets import build_multivariate_dataset_cae, build_synthetic_dataset_cae
from ..datasets import build_multivariate_dataset_cnn, build_synthetic_dataset_cnn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import silence_tensorflow
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


def build_report(model:Model, report:Callable, sequence:Sequence):
    predictions = model.predict_generator(
        sequence,
        steps=sequence.steps_per_epoch,
        verbose=1
    )
    y = np.concatenate([
        sequence[batch][1]
        for batch in tqdm(range(sequence.steps_per_epoch), desc="Concatenating outputs")
    ])
    return report(y, predictions)


def build_reports(**dataset_kwargs):
    reports = []
    for model_type in tqdm(models, desc="Model types", leave=False):
        report = report_types[model_type]
        for window_size, build_model in tqdm(models[model_type].items(), desc="Models", leave=False):
            training, validation = datasets[model_type]
            train, test = training(window_size, **dataset_kwargs)
            _, valid = validation(window_size, **dataset_kwargs)
            #bio = biological(window_size)
            model = build_model(verbose=False)
            model.load_weights(get_model_weights_path(model))
            reports += flat_report(
                build_report(model, report, test),
                model,
                training,
                "test"
            )
            reports += flat_report(
                build_report(model, report, train),
                model,
                training,
                "train"
            )
            reports += flat_report(
                build_report(model, report, valid),
                model,
                validation,
                "synthetic validation"
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
    pd.DataFrame(reports).to_csv("report.csv")
