from repairing_genomic_gaps import build_reports

def test_report():
    build_reports(
        root="test_reports",
        training_chromosomes=("chrM",),
        testing_chromosomes=("chrM",),
        batch_size=1
    )