# Centaur evaluation
Code for the paper "[Can Centaur Truly Simulate Human Cognition? The Fundamental Limitation of Instruction Understanding](https://y1ny.github.io/assets/centaur-evaluation.pdf)". 

See original Centaur [paper](https://www.nature.com/articles/s41586-025-09215-4) and [repo](https://github.com/marcelbinz/Llama-3.1-Centaur-70B/).

## Install & Getting Started

1. Clone the repository

2. Construct a virtual environment for this project

```bash
conda env create -f environment.yml
```

or you can install the packages using `requirements.txt`

```bash
pip install -r requirements.txt
```

3. Run the evaluation

```python
python evaluation.py
# change the model and test set path to you own.
```

**Notice**: The Centaur model and the test set should be downloaded following the instruction in the [original repo](https://github.com/marcelbinz/Llama-3.1-Centaur-70B).

## Citation

If you make use of the code in this repository, please cite the following papers:

