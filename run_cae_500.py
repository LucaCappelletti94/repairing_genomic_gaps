import silence_tensorflow
from notipy_me import Notipy
from repairing_genomic_gaps import cae_500, build_synthetic_dataset_cae, train_model

if __name__ == "__main__":
    with Notipy():
        model = cae_500()
        train, test = build_synthetic_dataset_cae(500, batch_size=512)
        model = train_model(model, train, test)
