from tensorflow.keras.utils import Sequence
from typing import List, Tuple, Dict
from ucsc_genomes_downloader import Genome
from ucsc_genomes_downloader.utils import tessellate_bed
from keras_synthetic_genome_sequence import SingleGapNoiseSequence, SingleGapSequence
import pandas as pd


def build_synthetic_dataset_sequence(
    window_size: int,
    keras_sequence_class: Sequence,
    assembly: str,
    chromosomes: List[str],
    batch_size: int,
    genome: Genome,
    seed: int
):
    """Return given keras sequence class for training and testing.

    Parameters
    --------------------------
    window_size: int,
        Windows size to extend gaps to.
    keras_sequence_class: Sequence,
        The class of Sequence to use to build the train and test sequences.
    assembly: str,
        Genomic assembly from which to extract sequences.
    chromosomes: List[str],
        List of chromosomes to use.
    batch_size: int
        Batch size for training the model.
    genome: Genome,
        Genome object from which to retrieve the filled regions
        for the given chromosomes.
    seed: int,
        Random seed to use when generating gaps.

    Returns
    ---------------------------
    Return Tuple of Sequences.
    """
    # Rendering genomic gaps
    return keras_sequence_class(
        assembly=assembly,
        bed=tessellate_bed(genome.filled(
            chromosomes=chromosomes
        ), window_size=window_size),
        batch_size=batch_size,
        seed=seed
    )


def build_synthetic_dataset(
    window_size: int,
    keras_sequence_class: Sequence,
    assembly: str = "hg19",
    training_chromosomes: List[str] = (
        "chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9",
        "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr19",
        "chr20", "chr21", "chr22", "chrX", "chrY"
    ),
    testing_chromosomes: List[str] = ("chr17", "chr18"),
    batch_size: int = 1024,
    seed: int = 42
) -> Tuple:
    """Return given keras sequence class for training and testing.

    Parameters
    --------------------------
    window_size: int,
        Windows size to extend gaps to.
    keras_sequence_class: Sequence,
        The class of Sequence to use to build the train and test sequences.
    assembly: str,
        Genomic assembly from which to extract sequences.
    training_chromosomes: List[str],
        List of chromosomes to use for training set.
    testing_chromosomes: List[str],
        List of chromosomes to use for test set.
    batch_size: int
        Batch size for training the model.
    seed: int,
        Random seed to use when generating gaps.

    Returns
    ---------------------------
    Return Tuple of Sequences.
    """
    # Retrieving and loading the assembly for required chromosomes

    genome = Genome(
        assembly=assembly,
        chromosomes=training_chromosomes+testing_chromosomes
    )

    training = build_synthetic_dataset_sequence(
        window_size,
        keras_sequence_class,
        assembly,
        training_chromosomes,
        batch_size,
        genome,
        seed
    )

    testing = build_synthetic_dataset_sequence(
        window_size,
        keras_sequence_class,
        assembly,
        testing_chromosomes,
        batch_size,
        genome,
        seed
    )

    return training, testing


def build_synthetic_dataset_cae(window_size:int, **kwargs:Dict)->Tuple[SingleGapNoiseSequence, SingleGapNoiseSequence]:
    """Return SingleGapNoiseSequence for training and testing.

    Parameters
    --------------------------
    window_size: int,
        Windows size to use for rendering the synthetic datasets.
    """
    return build_synthetic_dataset(window_size, SingleGapNoiseSequence, **kwargs)

def build_synthetic_dataset_cnn(window_size:int, **kwargs:Dict)->Tuple[SingleGapSequence, SingleGapSequence]:
    """Return SingleGapSequence for training and testing.

    Parameters
    --------------------------
    window_size: int,
        Windows size to use for rendering the synthetic datasets.
    """
    return build_synthetic_dataset(window_size, SingleGapSequence, **kwargs)