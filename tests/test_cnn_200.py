from repairing_genomic_gaps import cnn_200, build_synthetic_dataset_cnn, train_model

def test_cnn_200():
    model = cnn_200()
    model.summary()
    train, test = build_synthetic_dataset_cnn(
        200,
        training_chromosomes=["chr17"],
        testing_chromosomes=["chrM"],
        batch_size=8
    )
    train_model(model, train, test, epochs=1)

