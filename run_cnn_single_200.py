from notipy_me import Notipy
from repairing_genomic_gaps import cnn_200, build_synthetic_dataset_cnn, train_model


if __name__ == "__main__":
    with Notipy():
        model = cnn_200()
        train, test = build_synthetic_dataset_cnn(200)
        model = train_model(model, train, test, path="single_gap")