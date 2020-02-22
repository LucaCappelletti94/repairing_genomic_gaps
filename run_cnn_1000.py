import silence_tensorflow
from notipy_me import Notipy
from repairing_genomic_gaps import cnn_1000, build_synthetic_dataset_cnn, train_model


if __name__ == "__main__":
    with Notipy():
        model = cnn_1000()
        model.summary()
        train, test = build_synthetic_dataset_cnn(1000)
        model = train_model(model, train, test)