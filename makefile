
PYTHON_PATH=~/anaconda3/bin/

install:
	$(PYTHON_PATH)pip install --user --upgrade .

test:
	$(PYTHON_PATH)pytest -s --cov repairing_genomic_gaps --cov-report html

clear_cache:
	rm -rd ./cache/
	rm -rd ./genomes/
	rm -rd ./variables/

run:
	$(PYTHON_PATH)python run_cnn_200.py
	$(PYTHON_PATH)python run_cnn_500.py
	$(PYTHON_PATH)python run_cnn_1000.py
	$(PYTHON_PATH)python run_cae_200.py
	$(PYTHON_PATH)python run_cae_500.py
	$(PYTHON_PATH)python run_cae_1000.py

build:
	$(PYTHON_PATH)python setup.py sdist

publish:
	twine upload "./dist/$$(ls ./dist | grep .tar.gz | sort | tail -n 1)"