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
- Model saving and loading functionality

## Installation

1. Clone this repository:
   bash
   git clone https://github.com/softengunisalento/BertModelPII.git
   cd bert-pii-detection
   

2. Install the required dependencies:
   bash
   pip install -r requirements.txt
   

## Usage

### Training and Evaluation

1. Prepare your dataset in CSV format with 'Sentence' and 'Label' columns.

2. Run one of the training scripts:

   - Standard train-validation-test split:
     python
     python Main.py
     

   - K-fold cross-validation:
     python
     python Main.py  # Uncomment KFold_training(5,3) in Main.py
     

   - Stratified k-fold cross-validation:
     python
     python Main.py  # Uncomment Stratified_kfold_training(5,3) in Main.py
     

### Testing a Saved Model

To evaluate a saved model on a test dataset:
python
python testing_model.py


## File Structure


bert-pii-detection/
├── data/                    # Directory for training data
│   └── training_dataset.csv
├── Checkpoint/              # Directory for saved models
├── bert_model_trainer.py    # BERT model training and evaluation
├── data_handler.py          # Data loading and preprocessing
├── Main.py                  # Main training scripts
├── pii_data_loader.py       # Data loader for PII detection
├── requirements.txt         # Python dependencies
└── testing_model.py         # Script for testing saved models


## Contribution Guidelines

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Commit your changes
4. Push to the branch
5. Submit a pull request
