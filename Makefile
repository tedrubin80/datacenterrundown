.PHONY: install data idea3 idea5 all clean test

install:
	pip install -r requirements.txt

data:
	python3 pipelines/generate_data.py --rows 10000

idea3:
	python3 pipelines/train_idea3.py

idea5:
	python3 pipelines/train_idea5.py

all:
	python3 pipelines/full_pipeline.py

clean:
	rm -rf data/raw/* data/processed/* data/results/*

test:
	python3 -m pytest tests/ -v
