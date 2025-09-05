# Installation
Clone this repository using e.g.:
```bash
git clone https://github.com/centralflows/central_flows.git
```
We recommend working within a virtual environment.  In fact, we recommend using `uv`, which serves as a drop-in replacement for several Python package management tools including `pip` and `virtualenv`, but is much faster.
To install `uv`, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
To create a virtual environment, cd into `central_flows` and run:
```bash
uv venv --python=3.11
```
Here we use Python 3.11, but note that we've checked that the code works with Python 3.10 and 3.12 as well.

To activate the virtual environment, run:
```bash
source .venv/bin/activate
```
Then to install the required packages, run:
```bash
uv pip compile requirements.in -o requirements.txt 
uv pip install -r requirements.txt
```

By default, the code will save experiment data in an "experiments" subdirectory of the current working directory.  To use a different directory for storing experiment data, set the `EXPERIMENT_DIR` environment variable.
