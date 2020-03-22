from silence_tensorflow import silence_tensorflow
silence_tensorflow()

from .reports import build_reports

from .datasets import build_multivariate_dataset_cae, build_multivariate_dataset_cnn, build_synthetic_dataset_cae, build_synthetic_dataset_cnn
from .models import cnn_200, cnn_500, cnn_1000, cae_200, cae_500, cae_1000
from .train_model import train_model

__all__ = [
    "train_model",
    "build_multivariate_dataset_cae",
    "build_multivariate_dataset_cnn",
    "build_synthetic_dataset_cae",
    "build_synthetic_dataset_cnn",
    "cnn_200", "cnn_500", "cnn_1000",
    "cae_200", "cae_500", "cae_1000",
    "build_reports"
]
