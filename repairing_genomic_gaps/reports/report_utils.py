import numpy as np
from typing import Dict, List, Callable
from tensorflow.keras import Model
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score
from holdouts_generator.utils.metrics import binary_classifications_metrics
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
import math


def get_central_nucleotides(predictions: np.ndarray) -> np.ndarray:
    return predictions[:, predictions.shape[1]//2]


def categorical_report(y_true: np.ndarray, y_pred: np.ndarray, true_class: np.ndarray, pred_class: np.ndarray) -> Dict:
    return {
        "roc_auc_score": roc_auc_score(y_true, y_pred),
        "average_precision_score": average_precision_score(y_true, y_pred),
        "accuracy_score": accuracy_score(true_class, pred_class),
        "balanced_accuracy_score": balanced_accuracy_score(true_class, pred_class)
    }


def categorical_nucleotides_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    true_class = np.argmax(y_true, axis=-1)
    pred_class = np.argmax(y_pred, axis=-1)

    nucleotides = [
        "adenine",
        "cytosine",
        "thymine",
        "guanine"
    ]

    return {
        "all_nucleotides": categorical_report(
            y_true.flatten(), y_pred.flatten(),
            true_class.flatten(), pred_class.flatten()
        ),
        **{
            nucleotide: binary_classifications_metrics(
                y_true[:, :, i].flatten(), y_pred[:, :, i].flatten())
            for i, nucleotide in enumerate(nucleotides)
        }
    }


def cnn_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {
        "gap_filling": categorical_nucleotides_report(y_true.reshape(-1, 1, 4), y_pred.reshape(-1, 1, 4))
    }


def cae_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {
        "reconstruction": categorical_nucleotides_report(y_true, y_pred),
        **cnn_report(
            get_central_nucleotides(y_true),
            get_central_nucleotides(y_pred)
        )
    }


def _report_wrapper(kwargs):
    return kwargs["report"](kwargs["y_true"], kwargs["y_pred"])


def parallelize_report(report: Callable, y_true: np.ndarray, y_pred: np.ndarray):
    workers = min(10, cpu_count())
    total = workers*10
    chunk_size = math.ceil(y_true.shape[0]/total)
    tasks = (
        {
            "report":report,
            "y_true":y_true[chunk_size*i:(i+1)*chunk_size],
            "y_pred":y_pred[chunk_size*i:(i+1)*chunk_size]
        }
        for i in range(total)
    )
    with Pool(workers) as p:
        reports = list(tqdm(
            p.imap(_report_wrapper, tasks),
            total=total,
            desc="Computing {}".format(report.__name__)
        ))
        p.close()
        p.join()
    return reports


def flat_report(report: List[Dict], model: Model, dataset: Callable, run_type: str):
    return [
        {
            "model": model.name,
            "dataset": dataset.__name__,
            "task": task,
            "target": target,
            "run_type": run_type,
            **target_results
        }
        for report_set in report
        for task, results in report.items()
        for target, target_results in results.items()
    ]
