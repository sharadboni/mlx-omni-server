VENV := .venv
HOST ?= 0.0.0.0
PORT ?= 8765

.PHONY: install run-cached run clean stop

install:
	uv sync
	uv run python -m spacy download en_core_web_sm
	@echo "\nInstalled. Run 'make run' to start the server"

run: 
	uv run python -m server.app --host $(HOST) --port $(PORT)

run-cached: 
	uv run python -m server.app --host $(HOST) --port $(PORT) --keep-in-memory

stop:
	@lsof -ti:$(PORT) | xargs kill -9

clean:
	rm -rf $(VENV) __pycache__ server/__pycache__ server/routes/__pycache__
	@echo "Cleared"
