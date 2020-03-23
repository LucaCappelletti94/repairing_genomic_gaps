from notipy_me import Notipy
from repairing_genomic_gaps import build_reports


if __name__ == "__main__":
    with Notipy(task_name="Reports building"):
        build_reports("./reports")