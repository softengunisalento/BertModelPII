import torch
from transformers import BertForSequenceClassification, AdamW
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,roc_curve,auc


import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

class BertModelTrainer:
    def __init__(self, model_name, device, num_epochs=3, lr=5e-5, loss_fn=None):
        self.model_name = model_name
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_losses = []
        self.valid_accuracies = []
        
        # Se la funzione di perdita non è fornita, usa CrossEntropyLoss come default
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    # Altre funzioni della classe come train, evaluate, ecc. andranno qui


    def train(self, train_loader, valid_loader=None):
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.model.train()
            batch_losses = []

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                batch_losses.append(loss.item())

                if not batch_idx % 50:
                    print(f'Epoch: {epoch+1:04d}/{self.num_epochs:04d} | '
                        f'Batch {batch_idx:04d}/{len(train_loader):04d} | '
                        f'Loss: {loss:.4f}')

            # Calculate average loss for the epoch
            epoch_loss = sum(batch_losses) / len(batch_losses)
            self.train_losses.append(epoch_loss)

            # If valid_loader is provided, compute validation accuracy
            if valid_loader is not None:
                valid_accuracy = self.compute_accuracy(valid_loader)
                self.valid_accuracies.append(valid_accuracy)
                print(f'Epoch: {epoch+1:04d}/{self.num_epochs:04d} | '
                    f'Training Loss: {epoch_loss:.4f} | '
                    f'Validation Accuracy: {valid_accuracy:.2f}%')
            else:
                #kfolding 
                print(f'Epoch: {epoch+1:04d}/{self.num_epochs:04d} | '
                    f'Training Loss: {epoch_loss:.4f} | '
                    f'No validation step in this epoch.')

            print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')

        print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')

    def compute_accuracy(self, data_loader):
        self.model.eval()
        correct_pred, num_examples = 0, 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                _, predicted_labels = torch.max(outputs.logits, 1)
                num_examples += labels.size(0)
                correct_pred += (predicted_labels == labels).sum()
        
        accuracy = correct_pred.float() / num_examples * 100
        return accuracy.item()

    def plot_roc_curve(self, data_loader): 
        self.model.eval() 
        y_true = [] 
        y_scores = [] 
        with torch.no_grad(): 
            for batch in data_loader: 
                input_ids = batch['input_ids'].to(self.device) 
                attention_mask = batch['attention_mask'].to(self.device) 
                labels = batch['labels'].to(self.device) 
                outputs = self.model(input_ids, attention_mask=attention_mask) # Append true labels and predicted scores 
                y_true.extend(labels.cpu().numpy()) 
                y_scores.extend(outputs.logits.softmax(dim=1)[:, 1].cpu().numpy()) 
                # Prendiamo la probabilità della classe positiva # Calcola la curva ROC 



            fpr, tpr, _ = roc_curve(y_true, y_scores) 
            roc_auc = auc(fpr, tpr) # Plot della curva ROC con tutte le cifre significative 
            plt.figure(figsize=(10, 7)) 
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.5f})') # Mostra 5 cifre decimali 
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Aggiungere l'area riempita sotto la curva per l'effetto grafico desiderato 
            plt.fill_between(fpr, tpr, color='darkorange', alpha=0.3) 
            plt.xlabel('False Positive Rate (FPR)') 
            plt.ylabel('True Positive Rate (TPR)') 
            plt.title('Receiver Operating Characteristic (ROC) Curve') 
            plt.legend(loc='lower right') 
            plt.grid(True) 
            plt.show()
      

    def save_model(self, model_path, optimizer_path):
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optim.state_dict(), optimizer_path)

    def evaluate(self, test_loader):
            self.model.eval()  # Imposta il modello in modalità valutazione
            all_preds = []
            all_labels = []
            all_scores = []  # Per salvare i punteggi di probabilità per la curva ROC
            total_loss = 0


            with torch.no_grad():  # Disabilita il calcolo del gradiente
                for batch_index, batch in enumerate(test_loader):
                    inputs = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)


                    outputs = self.model(inputs, attention_mask=attention_mask)
                    logits = outputs.logits


                    # Calcolo loss
                    loss = self.loss_fn(logits, labels)
                    total_loss += loss.item()


                    # Previsioni e punteggi
                    _, preds = torch.max(logits, 1)
                    scores = torch.softmax(logits, dim=1)[:, 1]  # Prendi la probabilità per la classe positiva


                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_scores.extend(scores.cpu().numpy())


            # Metrice modello
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='weighted')
            recall = recall_score(all_labels, all_preds, average='weighted')
            f1 = f1_score(all_labels, all_preds, average='weighted')


            # AUC-ROC calcolato solo se ci sono più di una classe presente
            if len(set(all_labels)) > 1:
                auc_roc = roc_auc_score(all_labels, all_scores)
            
                # Calcolo dei falsi positivi e veri positivi per la curva ROC
                fpr, tpr, _ = roc_curve(all_labels, all_scores)


                # Visualizzazione della curva ROC
                plt.figure(figsize=(10, 7))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.5f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.fill_between(fpr, tpr, color='purple', alpha=0.2)  # Colora l'area sotto la curva
                plt.xlabel('False Positive Rate (FPR)')
                plt.ylabel('True Positive Rate (TPR)')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='lower right')
                plt.grid(True)
                plt.show()
            else:
                auc_roc = None


            avg_loss = total_loss / len(test_loader)


            # Output dei risultati
            print(f"Test Accuracy: {accuracy * 100:.5f}%")
            print(f"Precision: {precision:.5f}")
            print(f"Recall: {recall:.5f}")
            print(f"F1-Score: {f1:.5f}")
            if auc_roc is not None:
                print(f"AUC-ROC: {auc_roc:.5f}")
            else:
                print("AUC-ROC: Non calcolabile (meno di 2 classi)")
            print(f"Average Loss: {avg_loss:.4f}")


            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auc_roc": auc_roc,
                "loss": avg_loss
            }

    def load_model(self, model_path, optimizer_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)  # Move model to the correct device
        
        self.optim = AdamW(self.model.parameters(), lr=self.lr, correct_bias=True)
        
        try:
            optimizer_state_dict = torch.load(optimizer_path, map_location=self.device)
            #problema con il correct_bias(peso del modello),debug
            if 'state' in optimizer_state_dict and 'param_groups' in optimizer_state_dict:
                if 'correct_bias' not in optimizer_state_dict['param_groups'][0]:
                    print("The 'correct_bias' key is missing in the optimizer state dictionary.")
                self.optim.load_state_dict(optimizer_state_dict)
                print('Optimizer loaded successfully.')
            else:
                raise KeyError('Optimizer state dict is missing required keys.')
        except KeyError as e:
            print(f"Key error while loading optimizer state: {e}")
            print("Reinitializing optimizer.")
            self.optim = AdamW(self.model.parameters(), lr=self.lr, correct_bias=True)

    
    def plot_metrics(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.valid_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
