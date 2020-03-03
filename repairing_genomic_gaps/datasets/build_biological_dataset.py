from keras_biological_gaps_sequence import BiologicalGapsSequence
from cache_decorator import Cache

@Cache()
def build_biological_dataset_cnn(window_size:int, batch_size:int=20):
    return BiologicalGapsSequence(
        "hg19",
        "hg38",
        window_size,
        1,
        batch_size=batch_size
    )

@Cache()
def build_biological_dataset_cae(window_size:int, batch_size:int=20):
    return BiologicalGapsSequence(
        "hg19",
        "hg38",
        window_size,
        window_size,
        batch_size=batch_size
    )