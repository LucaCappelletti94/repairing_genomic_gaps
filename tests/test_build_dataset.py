from repairing_genomic_gaps import build_dataset


def test_build_dataset():
    max_gap_size = 100
    window_size = 500
    batch_size = 64
    train, test = build_dataset(
        assembly="hg19",
        training_chromosomes=["chr1"],
        testing_chromosomes=["chr18"],
        max_gap_size=max_gap_size,
        window_size=window_size,
        gaps_threshold=0.4,
        batch_size=batch_size
    )
    X_train, Y_train = train[0]
    X_test, Y_test = test[0]
    target_shape = (batch_size, window_size, 4)
    assert X_train.shape == Y_train.shape
    assert X_test.shape == Y_test.shape
    assert X_train.shape == target_shape
    assert X_test.shape == target_shape
