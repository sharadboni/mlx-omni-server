VENV := .venv
HOST ?= 0.0.0.0
PORT ?= 8765
PID_FILE ?= server.pid
LOG_FILE ?= server.log

.PHONY: install run run-cached run-bg run-bg-cached stop clean

install:
	uv sync
	uv pip install pip
	uv run python -m spacy download en_core_web_sm
	@echo "\nInstalled. Run 'make run' to start the server"

run:
	uv run python -m server.app --host $(HOST) --port $(PORT)

run-cached:
	uv run python -m server.app --host $(HOST) --port $(PORT) --keep-in-memory

run-bg:
	@nohup uv run python -m server.app --host $(HOST) --port $(PORT) > $(LOG_FILE) 2>&1 & echo $$! > $(PID_FILE)
	@echo "Server started (pid $$(cat $(PID_FILE))) — logs in $(LOG_FILE)"

run-bg-cached:
	@nohup uv run python -m server.app --host $(HOST) --port $(PORT) --keep-in-memory > $(LOG_FILE) 2>&1 & echo $$! > $(PID_FILE)
	@echo "Server started (pid $$(cat $(PID_FILE))) — logs in $(LOG_FILE)"

stop:
	@if [ -f $(PID_FILE) ]; then \
		kill $$(cat $(PID_FILE)) && rm $(PID_FILE) && echo "Server stopped"; \
	fi
	@lsof -ti:$(PORT),52415 | xargs kill -9 2>/dev/null || true

clean:
	rm -rf $(VENV) __pycache__ server/__pycache__ server/routes/__pycache__
	@echo "Cleared"
