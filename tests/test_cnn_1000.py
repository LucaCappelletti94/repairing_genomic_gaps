from repairing_genomic_gaps import cnn_1000, build_synthetic_dataset_cnn, train_model

def test_cnn_1000():
    model = cnn_1000()
    model.summary()
    train, test = build_synthetic_dataset_cnn(
        1000,
        training_chromosomes=["chrM"],
        testing_chromosomes=["chrM"],
        batch_size=128
    )
    train_model(model, train, test, epochs=1, path="./test_models")