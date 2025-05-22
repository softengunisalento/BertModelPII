from sklearn.model_selection import KFold
import pandas as pd
from sklearn.model_selection import StratifiedKFold

class DataHandler:
    def __init__(self, dataset_path, train_size=0.8, valid_size=0.1, test_size=0.1, n_splits=5):
        self.dataset_path = dataset_path
        self.df = None
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.n_splits = n_splits  # Add n_splits

    def load_data(self):
        self.df = pd.read_csv(self.dataset_path, encoding='utf-8', quotechar='"', skipinitialspace=True)
        print(self.df.columns)
        self.df['Sentence'] = self.df['Sentence'].astype(str)
    
    def clean_data(self):
        self.df['Sentence'] = self.df['Sentence'].str.strip().str.replace('.', '', regex=False).str.replace('"', '', regex=False)

    #Split data viene utilizzato per splittare i data per il training test eva
    def split_data(self):
        self.df = self.df.sample(frac=1, random_state=123).reset_index(drop=True)
        
        total_size = len(self.df)
        train_end = int(self.train_size * total_size)
        valid_end = train_end + int(self.valid_size * total_size)

        train_texts = self.df.iloc[:train_end]['Sentence'].tolist()
        train_labels = self.df.iloc[:train_end]['Label'].tolist()
        valid_texts = self.df.iloc[train_end:valid_end]['Sentence'].tolist()
        valid_labels = self.df.iloc[train_end:valid_end]['Label'].tolist()
        test_texts = self.df.iloc[valid_end:]['Sentence'].tolist()
        test_labels = self.df.iloc[valid_end:]['Label'].tolist()

        return train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels

    def kfold_split_data(self):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=123)
        sentences = self.df['Sentence'].tolist()
        labels = self.df['Label'].tolist()
        
        for train_index, test_index in kf.split(sentences):
            train_texts = [sentences[i] for i in train_index]
            train_labels = [labels[i] for i in train_index]
            test_texts = [sentences[i] for i in test_index]
            test_labels = [labels[i] for i in test_index]
            
            yield train_texts, train_labels, test_texts, test_labels


    def stratified_kfold_split_data(self):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=123)
        sentences = self.df['Sentence'].tolist()
        labels = self.df['Label'].tolist()

        for train_index, test_index in skf.split(sentences, labels):
            train_texts = [sentences[i] for i in train_index]
            train_labels = [labels[i] for i in train_index]
            test_texts = [sentences[i] for i in test_index]
            test_labels = [labels[i] for i in test_index]
            
            yield train_texts, train_labels, test_texts, test_labels
    
    def get_test_data(self):
        '''
        Utilizzata per prendere i dati per il processo di testing
        '''
        self.df = self.df.sample(frac=1, random_state=123).reset_index(drop=True)
        
     
        total_size = len(self.df)
        train_end = int(self.train_size * total_size)
        valid_end = train_end + int(self.valid_size * total_size)


        test_texts = self.df.iloc[valid_end:]['Sentence'].tolist()
        test_labels = self.df.iloc[valid_end:]['Label'].tolist()

        return test_texts, test_labels