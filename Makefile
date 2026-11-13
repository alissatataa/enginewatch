.PHONY: install test lint format train serve simulate docker-up docker-down

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

# Train the baseline on synthetic data (swap for --data-dir data/CMAPSS to use the real dataset)
train:
	enginewatch train --synthetic

serve:
	enginewatch serve --reload

simulate:
	enginewatch simulate --synthetic --units 5 --rate 25

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v
