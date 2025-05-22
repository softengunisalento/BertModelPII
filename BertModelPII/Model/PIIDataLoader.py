import torch
from transformers import BertTokenizerFast

class PIIDataLoader:
    def __init__(self, train_texts=None, train_labels=None, valid_texts=None, valid_labels=None, test_texts=None, test_labels=None,tokenizer_name='google-bert/bert-base-uncased',batch_size=64):
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.valid_texts = valid_texts if valid_texts is not None else []
        self.valid_labels = valid_labels if valid_labels is not None else []
        self.test_texts = test_texts
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        
    
    def tokenize_data(self, texts):
        if not texts:
            return None
        return self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

    def create_dataset(self, encodings, labels):
        return torch.utils.data.Dataset(encodings, labels)
    def get_specific_dataloader(self, mode):
        if mode == 'train':
            # Tokenizza i dati per il training
            train_encodings = self.tokenize_data(self.train_texts)
            train_dataset = PIIDataset(train_encodings, self.train_labels)
            return torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        elif mode == 'valid':
            # Tokenizza i dati per la validazione
            valid_encodings = self.tokenize_data(self.valid_texts)
            valid_dataset = PIIDataset(valid_encodings, self.valid_labels)
            return torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        elif mode == 'test':
            # Tokenizza i dati per il test
            test_encodings = self.tokenize_data(self.test_texts)
            test_dataset = PIIDataset(test_encodings, self.test_labels)
            return torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        else:
            raise ValueError("mode should be 'train', 'valid', or 'test'")

    def get_dataloader(self):
        train_encodings = self.tokenize_data(self.train_texts)
        valid_encodings = self.tokenize_data(self.valid_texts)
        test_encodings = self.tokenize_data(self.test_texts)

        train_dataset = PIIDataset(train_encodings, self.train_labels)
        valid_dataset = PIIDataset(valid_encodings, self.valid_labels)
        test_dataset = PIIDataset(test_encodings, self.test_labels)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader

class PIIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item



    def __len__(self):
        return len(self.labels)
