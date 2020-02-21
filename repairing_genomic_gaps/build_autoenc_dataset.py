
from .build_datasets import build_dataset

def build_autoenc_dataset(window_size, batch_size=2048):
    return build_dataset(
        assembly="hg19",
        training_chromosomes=[
            "chr1"
        ],
        testing_chromosomes=[
            "chr17"
        ],
        max_gap_size=1,
        window_size=window_size,
        gaps_threshold=0.4,
        batch_size=batch_size,
        seed=42
    )