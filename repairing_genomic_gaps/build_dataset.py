from typing import List, Tuple
from keras_synthetic_genome_sequence.utils import get_gaps_statistics
from ucsc_genomes_downloader import Genome
from ucsc_genomes_downloader.utils import tessellate_bed
from keras_synthetic_genome_sequence import GapSequence
from .utils import cache

@cache()
def build_dataset(
    assembly: str,
    training_chromosomes: List[str],
    testing_chromosomes: List[str],
    max_gap_size: int,
    window_size: int,
    gaps_threshold: float,
    batch_size: int,
    seed: int
) -> Tuple[GapSequence, GapSequence]:
    """Return keras GapSequences for training and testing.

    Parameters
    --------------------------
    assembly: str,
        Genomic assembly from which to extract sequences.
    training_chromosomes: List[str],
        List of chromosomes to use for training set.
    testing_chromosomes: List[str],
        List of chromosomes to use for test set.
    max_gap_size: int,
        Maximum gap size to use.
    window_size: int,
        Windows size to extend gaps to.
    gaps_threshold: float,
        Gaps 'activation' threshold from multivariate
        gaussian distribution.
    batch_size: int
        Batch size for training the model.
    seed: int,
        Random seed to use when generating gaps.

    Returns
    ---------------------------
    Return Tuple of GapSequences.
    """
    # Obtaining gaps statistic from given assembly
    number, mean, covariance = get_gaps_statistics(
        assembly=assembly,
        max_gap_size=max_gap_size,
        window_size=window_size
    )
    print("CULO", len(mean))
    print("Using {number} gaps for generating synthetic gaps.".format(
        number=number
    ))
    # Retrieving and loading the assembly for required chromosomes
    genome = Genome(
        assembly=assembly,
        chromosomes=training_chromosomes+testing_chromosomes
    )
    # Obtaining training bed file
    training_sequences = tessellate_bed(genome.filled(
        chromosomes=training_chromosomes
    ), window_size=window_size)
    # Obtaining testing bed file
    testing_sequences = tessellate_bed(genome.filled(
        chromosomes=testing_chromosomes
    ), window_size=window_size)
    # Rendering training genomic gaps
    training_gap_sequence = GapSequence(
        assembly=assembly,
        bed=training_sequences,
        gaps_mean=mean,
        gaps_covariance=covariance,
        gaps_threshold=gaps_threshold,
        batch_size=batch_size,
        seed=seed
    )
    # Rendering testing genomic gaps
    testing_gap_sequence = GapSequence(
        assembly=assembly,
        bed=testing_sequences,
        gaps_mean=mean,
        gaps_covariance=covariance,
        gaps_threshold=gaps_threshold,
        batch_size=batch_size,
        seed=seed
    )
    return training_gap_sequence, testing_gap_sequence
