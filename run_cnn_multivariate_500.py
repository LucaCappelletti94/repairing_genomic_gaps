import silence_tensorflow
from notipy_me import Notipy

from repairing_genomic_gaps import cnn_500, build_multivariate_dataset_cnn, train_model


if __name__ == "__main__":
    with Notipy():
        model = cnn_500()
        train, test = build_multivariate_dataset_cnn(500)
        model = train_model(model, train, test, path="multivariate_gaps")