# Deep RL Notebooks

A collection of Jupyter notebooks and Python code for learning and experimenting with deep reinforcement learning algorithms.

## Setup

```bash
uv install
uv run jupyter notebook --notebook-dir=notebooks/
```

## Project Structure

- `src/`: Python package with reusable components
  - `environments/`: Custom RL environments
  - `agents/`: RL algorithm implementations
  - `utils/`: Helper functions for visualization and data processing
- `notebooks/`: Jupyter notebooks for interactive learning
- `tests/`: Unit tests for the Python modules

## Usage

To import modules from the `src` package in your notebooks:

```python
# In your notebooks
from src.environments.grid_world import GridWorldEnv
from src.agents.value_iteration import ValueIterationAgent
from src.utils.plotting import plot_value_function

# Example usage
env = GridWorldEnv()
agent = ValueIterationAgent(env)
plot_value_function(agent.value_function)
```
