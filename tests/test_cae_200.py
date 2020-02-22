from repairing_genomic_gaps import cae_200, build_synthetic_dataset_cae, train_model

def test_cae_200():
    model = cae_200()
    model.summary()
    train, test = build_synthetic_dataset_cae(
        200,
        training_chromosomes=["chr17"],
        testing_chromosomes=["chrM"],
        batch_size=8
    )
    train_model(model, train, test, epochs=1, path="./test_models")

