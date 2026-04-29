# Code adapted from VerifierFC paper (Paper: https://arxiv.org/abs/2509.22101v1; Code: https://github.com/VenkteshV/VerifierFC).

import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_cosine_schedule_with_warmup
from utils import format_time, EarlyStopping, MultiTaskTextDataset, MultiTaskCustomClassifier, print_trainable_parameters
import time
import numpy as np 
import random
import os
from sklearn.metrics import accuracy_score
import tomli
import json

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


# Merge the toml config file with our default 
def merge_dicts(base, overrides):
    """
    Recursively merges two dictionaries.
    - Values in 'overrides' will overwrite or update values in 'base'.
    Used for rewriting the default config file with the values provided in toml config file.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            # Merge nested dictionaries
            base[key] = merge_dicts(base[key], value)
        elif not isinstance(value, dict):
            # Overwrite with non-dict or new key
            base[key] = value
    return base


class TrainerModule:
    def __init__(self, model, train_loader, val_loader, output_dir, epochs=3, lr = 2e-4, patience_value = 2, use_uwso = True):
        
        self.optimizer = AdamW(model.parameters(), lr = lr, eps = 1e-8)
        self.epochs = epochs
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.scoping_loss_func = torch.nn.BCEWithLogitsLoss()
        self.use_uwso = use_uwso
        self.uwso_temperature = 5.0 # larger temperature to allow even weighting
        self.uwso_eps = 1e-8
        
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.model = model
        self.model = model.to(self.device)
        
        # self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
        #     model, self.optimizer, train_loader, val_loader)
        
        
        
        
        self.output_dir = output_dir
        
        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(0.05 * total_steps)  # 5% warmup

        # Use cosine schedule with warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        
        self.early_stopping = EarlyStopping(patience=patience_value, verbose=True, path = self.output_dir)
        
    def compute_uwso_total_loss(self, loss, scoping_loss):
        losses_detached = torch.stack([loss.detach(), scoping_loss.detach()])

        inv_losses = 1.0 / (losses_detached + self.uwso_eps)
        weights = torch.softmax(inv_losses / self.uwso_temperature, dim=0)

        total_loss = weights[0] * loss + weights[1] * scoping_loss
        return total_loss, weights

    def train(self):
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()
        
        # For each epoch...
        for epoch_i in range(0, self.epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_accuracy = 0
            total_train_scoping_accuracy = 0
            total_train_loss = 0
            total_train_correctness_loss = 0
            total_train_scoping_loss = 0
            if self.use_uwso:
                total_train_correctness_weight = 0
                total_train_scoping_weight = 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questimport gensim.downloader as api
            self.model.train()

            # For each batch of training data...
            for step, batch in enumerate(self.train_loader):

                # Progress update every 40 batches.
                if step % 50 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_loader), elapsed))


                
                b_input_ids = batch['input_ids'].to(self.device)
                b_input_mask = batch['attention_mask'].to(self.device)
                b_labels = batch['labels'].to(self.device)
                b_scoping_labels = batch['scoping_labels'].to(self.device)

                self.optimizer.zero_grad()
                logits, scoping_logits = self.model(b_input_ids,b_input_mask)
                loss = self.loss_func(logits.squeeze(1), b_labels)
                scoping_loss = self.scoping_loss_func(scoping_logits.squeeze(1), b_scoping_labels)
                if self.use_uwso:
                    total_loss, weights = self.compute_uwso_total_loss(loss, scoping_loss) # compute loss with UW-SO
                    total_train_correctness_weight += weights[0].item()
                    total_train_scoping_weight += weights[1].item()
                else:
                    total_loss = loss + scoping_loss
                total_train_loss += total_loss.item()
                total_train_correctness_loss += loss.item()
                total_train_scoping_loss += scoping_loss.item()
                total_loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                
                probas = torch.sigmoid(logits)
                probas = probas.detach().cpu().numpy()
                pred_labels = (probas >= 0.5).astype(int)
                label_ids = b_labels.to('cpu').numpy()

                total_train_accuracy += accuracy_score(label_ids, pred_labels)

                probas = torch.sigmoid(scoping_logits)
                probas = probas.detach().cpu().numpy()
                pred_labels = (probas >= 0.5).astype(int)
                scoping_label_ids = b_scoping_labels.to('cpu').numpy()
                
                total_train_scoping_accuracy += accuracy_score(scoping_label_ids, pred_labels)
            
            avg_train_accuracy = total_train_accuracy / len(self.train_loader)
            
            print(" Train Accuracy: {0:.4f}".format(avg_train_accuracy))

            avg_train_scoping_accuracy = total_train_scoping_accuracy / len(self.train_loader)
            
            print(" Train Scoping Accuracy: {0:.4f}".format(avg_train_scoping_accuracy))

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(self.train_loader)
            avg_train_correctness_loss = total_train_correctness_loss / len(self.train_loader)
            avg_train_scoping_loss = total_train_scoping_loss / len(self.train_loader)

            # Measure average uwso weights
            if self.use_uwso:
                avg_train_correctness_weight = total_train_correctness_weight / len(self.train_loader)
                avg_train_scoping_weight = total_train_scoping_weight / len(self.train_loader)


            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.4f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))


            print("")
            print("Running Validation...")

            t0 = time.time()
            self.model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_scoping_accuracy = 0
            total_eval_loss = 0
            total_eval_correctness_loss = 0
            total_eval_scoping_loss = 0
            if self.use_uwso:
                total_eval_correctness_weight = 0
                total_eval_scoping_weight = 0

            # Evaluate data for one epoch
            for batch in self.val_loader:

                b_input_ids = batch['input_ids'].to(self.device)
                b_input_mask = batch['attention_mask'].to(self.device)
                b_labels = batch['labels'].to(self.device)
                b_scoping_labels = batch['scoping_labels'].to(self.device)
   
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    logits, scoping_logits = self.model(b_input_ids,b_input_mask)
                    
                    
                # Accumulate the validation loss.
                loss = self.loss_func(logits.squeeze(1), b_labels)
                scoping_loss = self.scoping_loss_func(scoping_logits.squeeze(1), b_scoping_labels)
                if self.use_uwso:
                    total_loss, weights = self.compute_uwso_total_loss(loss, scoping_loss) # compute loss with UW-SO
                    total_eval_correctness_weight += weights[0].item()
                    total_eval_scoping_weight += weights[1].item()
                else:
                    total_loss = loss + scoping_loss
                total_eval_loss += total_loss.item()
                total_eval_correctness_loss += loss.item()
                total_eval_scoping_loss += scoping_loss.item()

                # Move logits and labels to CPU
                probas = torch.sigmoid(logits)
                probas = probas.detach().cpu().numpy()
                pred_labels = (probas >= 0.5).astype(int)
                
                label_ids = b_labels.to('cpu').numpy()
                
                total_eval_accuracy += accuracy_score(label_ids, pred_labels)
                
                probas = torch.sigmoid(scoping_logits)
                probas = probas.detach().cpu().numpy()
                pred_labels = (probas >= 0.5).astype(int)
                
                scoping_label_ids = b_scoping_labels.to('cpu').numpy()

                total_eval_scoping_accuracy += accuracy_score(scoping_label_ids, pred_labels)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(self.val_loader)
            print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

            avg_val_scoping_accuracy = total_eval_scoping_accuracy / len(self.val_loader)
            print("  Scoping Accuracy: {0:.4f}".format(avg_val_scoping_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_correctness_loss = total_eval_correctness_loss / len(self.val_loader)
            avg_val_scoping_loss = total_eval_scoping_loss / len(self.val_loader)
            avg_val_loss = total_eval_loss / len(self.val_loader)
            self.early_stopping(avg_val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            # Measure average uwso weights
            if self.use_uwso:
                avg_val_correctness_weight = total_eval_correctness_weight / len(self.val_loader)
                avg_val_scoping_weight = total_eval_scoping_weight / len(self.val_loader)

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.4f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))
            output_dir = f"model_output/{self.output_dir}/"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print("Saving model to %s" % output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(self.model.state_dict(), os.path.join(output_dir, 'model_weights'))
            
            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Total Loss': avg_train_loss,
                    'Training Correctness Loss': avg_train_correctness_loss,
                    'Training Scoping Loss': avg_train_scoping_loss,
                    'Training Correctness Accur.': avg_train_accuracy,
                    'Training Scoping Accur.': avg_train_scoping_accuracy,
                    'Valid. Total Loss': avg_val_loss,
                    'Valid. Correctness Loss': avg_val_correctness_loss,
                    'Valid. Scoping Loss': avg_val_scoping_loss,
                    'Valid. Correctness Accur.': avg_val_accuracy,
                    'Valid. Scoping Accur.': avg_val_scoping_accuracy,
                    'Train Correctness Weight': avg_train_correctness_weight if self.use_uwso else 1.0,
                    'Train Scoping Weight': avg_train_scoping_weight if self.use_uwso else 1.0,
                    'Valid. Correctness Weight': avg_val_correctness_weight if self.use_uwso else 1.0,
                    'Valid. Scoping Weight': avg_val_scoping_weight if self.use_uwso else 1.0,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
        
        with open(f"model_output/{self.output_dir}/train_verbose.json", "w") as fp:
            json.dump(training_stats, fp, indent = 4)
        fp.close()
        
        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
        

if __name__ == "__main__":
            
    default_config = {
            "project name": "ORM verifier training",
            "version": "1.0.0",
            "configuration": {
                "train_data": "data/train_verifier/train_gpt_4o.csv",
                "val_data": "data/train_verifier/dev_gpt_4o.csv",
                "base_model": "meta-llama/Llama-3.2-3B",
                "is_base_encoder" : True, 
                "use_lora": False, 
                "lora_rank": 8, 
                "lora_alpha": 16, 
                "max_length": 1024,
                "batch_size": 16,
                "num_labels": 1,
                'patience': 2, 
                "lr": 1e-3,
                "hidden_dim": None, 
                "dropout_rate": 0,
                "num_epochs": 3,
                "output_dir": "default_name",
                "use_uwso": False,
                },
        }

    with open("code/train_verifier/config.toml", "rb") as fp:
        read_file = tomli.load(fp)
        user_config = read_file
    config = merge_dicts(default_config.copy(), user_config.copy())["configuration"]

    
    # Load Data
    train_df = pd.read_csv(config['train_data'])
    val_df = pd.read_csv(config['val_data'])
    
    # Initialize tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    
    if config['is_base_encoder'] == False:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = MultiTaskTextDataset(train_df, tokenizer, max_length=config['max_length'])
    val_dataset = MultiTaskTextDataset(val_df, tokenizer, max_length=config['max_length'])
    
    train_loader = DataLoader(
            train_dataset,  # The training samples.
            batch_size = config['batch_size'] # Trains with this batch size.
        )

    val_loader = DataLoader(
            val_dataset, # The validation samples.
            batch_size = config['batch_size']
        )
    
    
    # Initialize model. For a binary classification put output units/neurons as 1.
    model = MultiTaskCustomClassifier(config['base_model'], num_labels= config['num_labels'], hidden_dim=config['hidden_dim'], dropout_value=config['dropout_rate'], 
                             use_lora= config['use_lora'], is_base_encoder= config['is_base_encoder'], lora_rank = config['lora_rank'], lora_alpha = config['lora_alpha'])

    print_trainable_parameters(model)
    
    # Initialize trainer
    trainer = TrainerModule(model, train_loader, val_loader,  output_dir= config['output_dir'], 
                            epochs=config['num_epochs'], lr = config['lr'], patience_value=config['patience'], use_uwso=config['use_uwso'])
    
    
    # Start training
    trainer.train()

    ### Train second model with uwso!
    with open("code/train_verifier/config_uwso.toml", "rb") as fp:
        read_file = tomli.load(fp)
        user_config = read_file
    config = merge_dicts(default_config.copy(), user_config.copy())["configuration"]

    
    # Load Data
    train_df = pd.read_csv(config['train_data'])
    val_df = pd.read_csv(config['val_data'])
    
    # Initialize tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    
    if config['is_base_encoder'] == False:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = MultiTaskTextDataset(train_df, tokenizer, max_length=config['max_length'])
    val_dataset = MultiTaskTextDataset(val_df, tokenizer, max_length=config['max_length'])
    
    train_loader = DataLoader(
            train_dataset,  # The training samples.
            batch_size = config['batch_size'] # Trains with this batch size.
        )

    val_loader = DataLoader(
            val_dataset, # The validation samples.
            batch_size = config['batch_size']
        )
    
    
    # Initialize model. For a binary classification put output units/neurons as 1.
    model = MultiTaskCustomClassifier(config['base_model'], num_labels= config['num_labels'], hidden_dim=config['hidden_dim'], dropout_value=config['dropout_rate'], 
                             use_lora= config['use_lora'], is_base_encoder= config['is_base_encoder'], lora_rank = config['lora_rank'], lora_alpha = config['lora_alpha'])

    print_trainable_parameters(model)
    
    # Initialize trainer
    trainer = TrainerModule(model, train_loader, val_loader,  output_dir= config['output_dir'], 
                            epochs=config['num_epochs'], lr = config['lr'], patience_value=config['patience'], use_uwso=config['use_uwso'])
    
    
    # Start training
    trainer.train()
