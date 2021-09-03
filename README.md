# Sampling-free Laplace Approximation for Bayesian Neural Network
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Pytorch 1.9.0](https://img.shields.io/badge/pytorch-1.9.0-blue.svg)](https://pytorch.org/)

PyTorch implementation of the following methods:
- ‘In-Between’ Uncertainty in Bayesian Neural Networks ([link](https://arxiv.org/pdf/1906.11537.pdf))
- A Scalable Laplace Approximation for Neural Networks ([link](https://openreview.net/pdf?id=Skdvd2xAZ))

## Requirements
- Python 3.6.0
- PyTorch 1.9.0
- Numpy 1.18.1 
- Matplotlib 3.1.2
- tqdm 4.42.0 

This code can run on either CUDA or CPU, which depends on your environments.


## Structure
### Regression experiments

We carried out homoscedastic and heteroscedastic regression experiements on toy datasets, generated with (Gaussian Process ground truth), as well as on real data (six UCI datasets).

Notebooks/classification/(ModelName)_(ExperimentType).ipynb: Contains experiments using (ModelName) on (ExperimentType), i.e. homoscedastic/heteroscedastic. The heteroscedastic notebooks contain both toy and UCI dataset experiments for a given (ModelName).

We also provide Google Colab notebooks. This means that you can run on a GPU (for free!). No modifications required - all dependencies and datasets are added from within the notebooks - except for selecting Runtime -> Change runtime type -> Hardware accelerator -> GPU.

### MNIST classification experiments

train_(ModelName)_(Dataset).py: Trains (ModelName) on (Dataset). Training metrics and model weights will be saved to the specified directories.

src/: General utilities and model definitions.

Notebooks/classification: An asortment of notebooks which allow for model training, evaluation and running of digit rotation uncertainty experiments. They also allow for weight distribution plotting and weight pruning. They allow for loading of pre-trained models for experimentation.




## Installation
Installation Instructions. Example: 

*Clone repository and install Python dependencies.*
```sh
$ git clone https://github.com/ZiqingZhao/Sampling-free-BNN.git
$ conda create -n your_venv
$ conda activate your_venv
$ conda install --file requirements.txt 
```
*If using pip, change "opencv" to "opencv-python" and "=" to "==":*

```sh
$ pip3 install -r requirements.txt 
```

## Setup local data directory
How to setup the dataset directory in order to run training/testing scripts. Example:
```sh
$ mkdir your_data_dir/kia/tranche3/xyz
...
```

## Training (if applicable)
Example:
```sh
$ python your_train_script.py --train_dir path/to/dir
...
```

Please also state the most important args to run the script.

## Testing
Example:
```sh
$ python your_test_script.py --train_dir path/to/dir
...
```
State where the validation results are being saved.
State how to run a visualization (if applicable).
State the most important args to run the script.

## Example Results
Show some results from the testing scripts (visualizations, metric outputs etc.) so the viewer becomes a feeling for the input/output of the method.

## Contributors
Name, Company, Email

## Contact
Name, Company, Email
