install:
	python3 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"

test:
	. .venv/bin/activate && pytest

lint:
	. .venv/bin/activate && ruff check .

format:
	. .venv/bin/activate && ruff format .

train-baseline:
	.venv/bin/python -m plant_disease_visionops.training.train_baseline \
		--raw-data-dir data/raw \
		--processed-dir data/processed \
		--out-dir artifacts/models/baseline_cnn \
		--reports-dir reports \
		--figures-dir artifacts/figures \
		--image-size 128 \
		--batch-size 16 \
		--epochs 3 \
		--learning-rate 0.001 \
		--num-workers 2 \
		--seed 42
