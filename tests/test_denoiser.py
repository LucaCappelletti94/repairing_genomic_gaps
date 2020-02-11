from repairing_genomic_gaps import build_dataset, build_denoiser
from keras_tqdm import TQDMCallback
from multiprocessing import cpu_count


def test_build_dataset():
    max_gap_size = 10
    window_size = 40
    batch_size = 8
    train, test = build_dataset(
        assembly="hg19",
        training_chromosomes=["chr17"],
        testing_chromosomes=["chrM"],
        max_gap_size=max_gap_size,
        window_size=window_size,
        gaps_threshold=0.4,
        batch_size=batch_size,
        seed=42
    )
    X_train, Y_train = train[0]
    X_test, Y_test = test[0]
    target_shape = (batch_size, window_size, 4)
    assert X_train.shape == Y_train.shape
    assert X_test.shape == Y_test.shape
    assert X_train.shape == target_shape
    assert X_test.shape == target_shape

    model = build_denoiser(window_size)

    model.fit_generator(
        train,
        steps_per_epoch=2,
        epochs=2,
        shuffle=True,
        verbose=0,
        callbacks=[
            TQDMCallback()
        ],
        validation_data=test,
        validation_steps=2,
        workers=cpu_count()//2,
        use_multiprocessing=True
    )
