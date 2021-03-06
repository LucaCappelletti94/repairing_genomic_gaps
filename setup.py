import os
import re
# To use a consistent encoding
from codecs import open as copen

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the relevant file
with copen(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


def read(*parts):
    with copen(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


__version__ = find_version("repairing_genomic_gaps", "__version__.py")

test_deps = [
    "pytest",
    "pytest-cov",
    "coveralls",
    "validate_version_code",
    "codacy-coverage",
    "keras_tqdm",
    "notipy_me"
]

extras = {
    'test': test_deps,
}

setup(
    name='repairing_genomic_gaps',
    version=__version__,
    description="Experiment on the possibility of repairing genomic gaps using auto-encoders.",
    long_description=long_description,
    url="https://github.com/LucaCappelletti94/repairing_genomic_gaps",
    author="Luca Cappelletti, Tommaso Fontana",
    author_email="cappelletti.luca94@gmail.com",
    # Choose your license
    license='MIT',
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    tests_require=test_deps,
    # Add here the package dependencies
    install_requires=[
        "keras_synthetic_genome_sequence>=1.0.9",
        "dict_hash",
        "plot_keras_history>=1.1.19",
        "cache_decorator>=1.1.0"
        "deflate_dict",
        "silence_tensorflow>=1.1.0",
        "keras_biological_gaps_sequence"
    ],
    extras_require=extras,
)
