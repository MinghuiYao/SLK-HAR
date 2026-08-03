# Contributing

Install development dependencies with `python -m pip install -e ".[dev]"` and
run `pytest`, `ruff check .`, and `ruff format --check .`. Sparse-training
changes must test mask device placement, density conservation, and optimizer
state masking. Do not commit datasets, checkpoints, generated runs, or secrets.
