.PHONY: help install install-gpu smoke tiny-gpu-smoke test lint typecheck lock check clean

help:
	@printf '%s\n' 'make install | smoke | test | lint | typecheck | check'

install:
	uv sync --extra dev

install-gpu:
	uv sync --extra data --extra gpu --extra eval --extra tracking --extra dev

smoke:
	bash scripts/smoke_pipeline.sh

tiny-gpu-smoke:
	bash scripts/tiny_gpu_smoke.sh

test:
	PYTHONPATH=src uv run --no-sync python -m unittest discover -s tests -v

lint:
	uv run --no-sync ruff check src tests
	uv run --no-sync ruff format --check src tests

typecheck:
	uv run --no-sync mypy src

lock:
	uv lock --check

check: test lint typecheck lock

clean:
	rm -rf artifacts/smoke .pytest_cache .ruff_cache .mypy_cache
