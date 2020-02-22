from repairing_genomic_gaps import cae_500, build_synthetic_dataset_cae, train_model

def test_cae_500():
    model = cae_500()
    train, test = build_synthetic_dataset_cae(
        500,
        training_chromosomes=["chrM"],
        testing_chromosomes=["chrM"],
        batch_size=128
    )
    train_model(model, train, test, epochs=1, path="./test_models")

