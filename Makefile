.PHONY: help train eval test dashboard clean setup

PYTHON := python
INTERSECTION := configs/intersection_casablanca.yaml
TRAINING_CFG := configs/training.yaml

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Install Python dependencies
	pip install -r requirements.txt

train:  ## Run PPO training (headless)
	$(PYTHON) training/train.py --intersection $(INTERSECTION) --config $(TRAINING_CFG)

train-gui:  ## Run PPO training with SUMO GUI
	$(PYTHON) training/train.py --intersection $(INTERSECTION) --config $(TRAINING_CFG) --gui

eval:  ## Evaluate PPO agent vs FixedCycle baseline
	$(PYTHON) training/evaluate.py --intersection $(INTERSECTION) --config $(TRAINING_CFG)

test:  ## Run unit tests (no SUMO required)
	$(PYTHON) -m pytest tests/ -v --tb=short

dashboard:  ## Launch real-time Streamlit dashboard
	streamlit run dashboard/app.py

routes:  ## Regenerate SUMO route files
	$(PYTHON) sumo/tools/generate_routes.py

clean:  ## Remove generated files (models, logs, cache)
	rm -rf models/ logs/ checkpoints/ __pycache__ */__pycache__ .pytest_cache

lint:  ## Run linter
	$(PYTHON) -m flake8 env/ agents/ training/ utils/ dashboard/ --max-line-length=100

.DEFAULT_GOAL := help
