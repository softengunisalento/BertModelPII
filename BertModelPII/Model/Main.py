import torch
from DataHandler import DataHandler
from PIIDataLoader import PIIDataLoader
from BertModelTrainer import BertModelTrainer

def training_test_validation(NUM_EPOCH):
    torch.backends.cudnn.deterministic = True
    RANDOM_SEED = 123
    torch.manual_seed(RANDOM_SEED)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NUM_EPOCHS = NUM_EPOCH 

    # Load and preprocess data
    dataset_path = 'training_dataset.csv'

    data_handler = DataHandler(dataset_path)
    data_handler.load_data()
    data_handler.clean_data()
    train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels = data_handler.split_data()

    # Create data loaders
    data_loader = PIIDataLoader(train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels)
    train_loader, valid_loader, test_loader = data_loader.get_dataloader()

    # Train and evaluate model
    model_trainer = BertModelTrainer(model_name='google-bert/bert-base-uncased', device=DEVICE, num_epochs=NUM_EPOCHS)
    model_trainer.train(train_loader, valid_loader)
    model_trainer.evaluate(test_loader)

    # Save the model
    model_trainer.save_model('bert_model_tte.pt', 'optimizer_tte.pt')
    #torch.save(model_trainer, 'CheckPoint/tensor.pt')
    model_trainer.plot_metrics()
    
def KFold_training(N_SPLIT,NUM_EPOCH):
    torch.backends.cudnn.deterministic = True
    RANDOM_SEED = 123
    torch.manual_seed(RANDOM_SEED)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    NUM_EPOCHS = NUM_EPOCH
    N_SPLITS = N_SPLIT
    dataset_path = 'training_dataset.csv'
    data_handler = DataHandler(dataset_path, n_splits=N_SPLITS)
    data_handler.load_data()
    data_handler.clean_data()

    fold_results = []

    for fold, (train_texts, train_labels, test_texts, test_labels) in enumerate(data_handler.kfold_split_data()):
        print(f"Training on fold {fold + 1}/{N_SPLITS}...")

        data_loader = PIIDataLoader(train_texts, train_labels, [], [], test_texts, test_labels)
        train_loader, _, test_loader = data_loader.get_dataloader()

        model_trainer = BertModelTrainer(model_name='google-bert/bert-base-uncased', device=DEVICE, num_epochs=NUM_EPOCHS)


        model_trainer.train(train_loader, None) 

        test_results = model_trainer.evaluate(test_loader)
        fold_results.append(test_results)  

        model_trainer.save_model(f'Checkpoint/bert_model_fold_{fold + 1}.pt', f'Checkpoint/optimizer_fold_{fold + 1}.pt')

    print(f"Cross-validation completed. Results: {fold_results}")


    model_trainer.plot_metrics()

def Stratified_kfold_training(N_SPLIT, NUM_EPOCH):
    torch.backends.cudnn.deterministic = True
    RANDOM_SEED = 123
    torch.manual_seed(RANDOM_SEED)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    NUM_EPOCHS = NUM_EPOCH
    N_SPLITS = N_SPLIT
    dataset_path = 'training_dataset.csv'
    data_handler = DataHandler(dataset_path, n_splits=N_SPLITS)
    data_handler.load_data()
    data_handler.clean_data()

    fold_results = []

  
    for fold, (train_texts, train_labels, test_texts, test_labels) in enumerate(data_handler.stratified_kfold_split_data()):
        print(f"Training on stratified fold {fold + 1}/{N_SPLITS}...")

   
        data_loader = PIIDataLoader(train_texts, train_labels, [], [], test_texts, test_labels)
        train_loader, _, test_loader = data_loader.get_dataloader()

       
        model_trainer = BertModelTrainer(model_name='google-bert/bert-base-uncased', device=DEVICE, num_epochs=NUM_EPOCHS)

      
        model_trainer.train(train_loader, None)  

       
        test_results = model_trainer.evaluate(test_loader)
        fold_results.append(test_results)  

       
        model_trainer.save_model(f'Checkpoint/bert_model_Strfold_{fold + 1}.pt', f'Checkpoint/optimizer_Strfold_{fold + 1}.pt')

    print(f"Stratified Cross-validation completed. Results: {fold_results}")

    model_trainer.plot_metrics()




training_test_validation(3)
#KFold_training(5,3)
#Stratified_kfold_training(5,3)


    
