
from .build_datasets import build_dataset

def build_autoenc_dataset(window_size, batch_size=2048):
    return build_dataset(
        assembly="hg19",
        training_chromosomes=[
            "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
            "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr19",
            "chr20", "chr21", "chr22", "chrX", "chrY"
        ],
        testing_chromosomes=[
            "chr17",
            "chr18"
        ],
        max_gap_size=1,
        window_size=window_size,
        gaps_threshold=0.4,
        batch_size=batch_size,
        seed=42
    )