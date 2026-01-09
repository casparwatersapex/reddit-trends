SHELL := /bin/bash

.PHONY: install lint format test app

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

test:
	pytest -q

app:
	streamlit run src/client_portal/app/streamlit_app.py
