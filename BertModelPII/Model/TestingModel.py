import torch
import pandas as pd
from DataHandler import DataHandler
from PIIDataLoader import PIIDataLoader
from BertModelTrainer import BertModelTrainer

def load_and_evaluate_model(model_path, optimizer_path, dataset_path):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Caricamento del dataset
    data_handler = DataHandler(dataset_path)
    data_handler.load_data()
    data_handler.clean_data()



    test_texts, test_labels = data_handler.get_test_data()


    #vuoti in quando prende i dati solo del testing
    data_loader = PIIDataLoader([], [], [], [], test_texts, test_labels)
    test_loader = data_loader.get_specific_dataloader(mode='test')

    model_trainer = BertModelTrainer(model_name='google-bert/bert-base-uncased', device=DEVICE)
    model_trainer.load_model(model_path, optimizer_path)

    model_trainer.evaluate(test_loader)
    #model_trainer.plot_roc_curve(test_loader)


model_path = 'Checkpoint/bert_model_Strfold_5.pt'
optimizer_path = 'Checkpoint/optimizer_Strfold_5.pt'
dataset_path = 'testing_dataset.csv' 
load_and_evaluate_model(model_path, optimizer_path, dataset_path)


