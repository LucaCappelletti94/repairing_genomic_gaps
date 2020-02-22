from repairing_genomic_gaps import cnn_500, build_synthetic_dataset_cnn, train_model

def test_cnn_500():
    model = cnn_500()
    train, test = build_synthetic_dataset_cnn(
        500,
        training_chromosomes=["chrM"],
        testing_chromosomes=["chrM"],
        batch_size=128
    )
    train_model(model, train, test, epochs=1, path="./test_models")