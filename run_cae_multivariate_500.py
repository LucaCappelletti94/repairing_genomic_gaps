from notipy_me import Notipy
from repairing_genomic_gaps import cae_500, build_multivariate_dataset_cae, train_model

if __name__ == "__main__":
    with Notipy():
        model = cae_500()
        train, test = build_multivariate_dataset_cae(500)
        model = train_model(model, train, test, path="multivariate_gaps")
