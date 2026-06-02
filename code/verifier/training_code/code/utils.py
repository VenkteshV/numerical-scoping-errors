# Code adapted from VerifierFC paper (Paper: https://arxiv.org/abs/2509.22101v1; Code: https://github.com/VenkteshV/VerifierFC).

import numpy as np
import datetime
import torch
from torch.utils.data import Dataset
from transformers import AutoModel
import os
from sklearn.metrics import classification_report, confusion_matrix,f1_score
import evaluate
from peft import LoraConfig, get_peft_model

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
    
class CustomClassifier(torch.nn.Module):
    """
    Load the base LM and add a single classification head on top of it.
    """
    def __init__(self, model_name, num_labels=2, hidden_dim = None, dropout_value = 0.1, freeze_base_layer = True, use_lora = False, 
                 is_base_encoder = True, lora_rank = 8, lora_alpha = 16):
        
        
        super(CustomClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.is_base_encoder = is_base_encoder
        
        if freeze_base_layer:
            print("Freezing the base layers")
            for param in self.model.parameters():
                param.requires_grad = False
        
        if use_lora:
            print("Using LoRa for fine tuning")
            lora_config = LoraConfig(
                r = lora_rank, # The rank of the adapters
                lora_alpha = lora_alpha, # Scaling factor
                target_modules= ['q_proj', 'k_proj', 'v_proj'],
                lora_dropout=0,
                bias = 'none'
            )
            
            self.model = get_peft_model(self.model, lora_config)
        
        
        if hidden_dim is not None:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.model.config.hidden_size, hidden_dim),
                torch.nn.ReLU(), 
                torch.nn.Dropout(dropout_value),
                torch.nn.Linear(hidden_dim, num_labels)
            )
        elif hidden_dim is None:
            self.classifier = torch.nn.Linear(self.model.config.hidden_size, num_labels)
        
        

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        
        if self.is_base_encoder:
            pooled_output = outputs.pooler_output
        else: 
            pooled_output = outputs.last_hidden_state[:, -1, :]
        logits = self.classifier(pooled_output)
        return logits


class MultiTaskCustomClassifier(torch.nn.Module):
    """
    Load the base LM and add two classification heads on top of it.
    """
    def __init__(self, model_name, num_labels=2, hidden_dim = None, dropout_value = 0.1, freeze_base_layer = True, use_lora = False, 
                 is_base_encoder = True, lora_rank = 8, lora_alpha = 16):
        
        
        super(MultiTaskCustomClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.is_base_encoder = is_base_encoder
        
        if freeze_base_layer:
            print("Freezing the base layers")
            for param in self.model.parameters():
                param.requires_grad = False
        
        if use_lora:
            print("Using LoRa for fine tuning")
            lora_config = LoraConfig(
                r = lora_rank, # The rank of the adapters
                lora_alpha = lora_alpha, # Scaling factor
                target_modules= ['q_proj', 'k_proj', 'v_proj'],
                lora_dropout=0,
                bias = 'none'
            )
            
            self.model = get_peft_model(self.model, lora_config)
        
        
        if hidden_dim is not None:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.model.config.hidden_size, hidden_dim),
                torch.nn.ReLU(), 
                torch.nn.Dropout(dropout_value),
                torch.nn.Linear(hidden_dim, num_labels)
            )
            self.scoping_classifier = torch.nn.Sequential(
                torch.nn.Linear(self.model.config.hidden_size, hidden_dim),
                torch.nn.ReLU(), 
                torch.nn.Dropout(dropout_value),
                torch.nn.Linear(hidden_dim, num_labels)
            )
        elif hidden_dim is None:
            self.classifier = torch.nn.Linear(self.model.config.hidden_size, num_labels)
            self.scoping_classifier = torch.nn.Linear(self.model.config.hidden_size, num_labels)
        
        

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        
        if self.is_base_encoder:
            pooled_output = outputs.pooler_output
        else: 
            pooled_output = outputs.last_hidden_state[:, -1, :]
        logits = self.classifier(pooled_output)
        scoping_logits = self.scoping_classifier(pooled_output)
        return logits, scoping_logits


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='model_output', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = f'model_output/{path}/'
        self.trace_func = trace_func
        
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(model.state_dict(), f"{self.path}/checkpoint.pt")
        self.val_loss_min = val_loss
        
      
class TextDataset(Dataset):
    
    """Load training dataset. 
    """
    def __init__(self, dataframe, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = dataframe['input_text'].tolist()
        self.labels = dataframe['Class'].tolist()
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
   
class ScopingTextDataset(Dataset):
    """
    Load scoping training dataset. 
    """
    def __init__(self, dataframe, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = dataframe['input_text'].tolist()
        self.labels = dataframe['Class_scoping'].tolist()
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
        
      
class MultiTaskTextDataset(Dataset):
    """
    Load multi-task training dataset. 
    """
    def __init__(self, dataframe, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = dataframe['input_text'].tolist()
        self.labels = dataframe['Class'].tolist()
        self.scoping_labels = dataframe['Class_scoping'].tolist()
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        item['scoping_labels'] = torch.tensor(self.scoping_labels[idx], dtype=torch.float)
        return item


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))