# BERT-based Text Classification for PII Detection

This project provides a toolkit for training and evaluating BERT-based models for text classification tasks, specifically designed for Personally Identifiable Information (PII) detection. It includes utilities for data handling, model training, and evaluation with support for standard training-validation-test splits as well as k-fold and stratified k-fold cross-validation.

## Features

- BERT model training and evaluation
- Support for multiple data splitting strategies:
  - Standard train-validation-test split
  - K-fold cross-validation
  - Stratified k-fold cross-validation
- Comprehensive evaluation metrics including accuracy, precision, recall, F1-score
- Visualization of training metrics

## Installation

Clone this repository:
```bash
git clone https://github.com/softengunisalento/BertModelPII.git
cd BertModelPII
```

Create a venv and activate it:
```bash
python3 -m venv .env
source .env/bin/activate
```
> To deactivate the venv use the `deactivate` command in the terminal

Install the required dependencies:

```bash
pip3 install -r requirements.txt
```

## Usage

### Training and Evaluation

To train BERT model:
```bash
cd models
python3 training.py 
```

this will run one between:

- Standard train-validation-test split:
```bash
python3 training.py --standard
```

- K-fold cross-validation:
```bash
python3 training.py --kfold
```

- Stratified k-fold cross-validation:
```bash
python3 training.py --strkfold
```

> Note that u can change the number of splits and epochs using the flags:
> - splits number: default = 5
> - epochs number: default = 3
> 
> example with default parameters:
> ```bash
> python3 training.py -b
> ```
> 
> example with changed parameters:
> ```bash
> python3 training.py -b -e 5 -n 8 --dataset datasets/prova.py
> ```

### Testing a Saved Model

To evaluate a saved model on a test dataset:
```bash
python3 testing.py --results_path "" --model_path "" --dataset_path "" --optimizer_path ""
```

## File Structure

```
BertModelPII/
├── custom_libs/                # Custom functions
│   ├── bert_model_trainer.py   # BERT model training and evaluation
│   ├── data_handler.py         # Data loading and preprocessing
│   └── pii_data_loader.py      # Data loader for PII detection
├── datasets/                   # Directory for datasets
│   ├── testing_dataset.csv
│   └── training_dataset.csv
├── docs/                       # Directory for thesis and paper pdfs
├── thesis_results/             # Directory for "right" results
│   ├── testing/
│   │   └── ...
│   └── training
│       └── ...
├── requirements.txt            # Python dependencies
├── testing.py                  # Script for testing saved models
└── training.py                 # Main training scripts
```
