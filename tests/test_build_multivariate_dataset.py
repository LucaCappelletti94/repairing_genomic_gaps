from repairing_genomic_gaps import build_multivariate_dataset_cae, build_multivariate_dataset_cnn


def test_build_multivariate_dataset():
    build_multivariate_dataset_cae(
        1000,
        training_chromosomes=["chrM"],
        testing_chromosomes=["chrM"]
    )
    build_multivariate_dataset_cnn(
        1000,
        training_chromosomes=["chrM"],
        testing_chromosomes=["chrM"]
    )
