# Lightning JTML (LitJTML)

### This repo is for moving our JTML neural network code to PyTorch Lightning and using WandB logging.



## Setup:

1. Create the conda environment `lit-jtml-env` from the `environment.yml` using the command `conda create env create -f environment.yml`.
2. Activate the conda env with `conda activate lit-jtml-env`.

## Use:

1. Be in the LitJTML directory (use the `cd` command to change the directory to the `blah/blah/LitJTML/` directory).
2. To fit (train) a model, call `python scripts/fit.py my_config` where `my_config` is the name of the config.py file in the `config/` directory.
    - The config file should specify the model, data, and other parameters.
