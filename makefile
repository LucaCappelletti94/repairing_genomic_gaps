
PYTHON_PATH=~/anaconda3/bin/

install:
	$(PYTHON_PATH)pip install --user --upgrade .

test:
	$(PYTHON_PATH)pytest -s --cov repairing_genomic_gaps --cov-report html

build:
	$(PYTHON_PATH)python setup.py sdist

publish:
	twine upload "./dist/$$(ls ./dist | grep .tar.gz | sort | tail -n 1)"