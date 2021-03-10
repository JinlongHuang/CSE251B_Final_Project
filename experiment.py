################################################################################
# CSE 251 B: Final Project
# Code by Keshav Rungta, Geeling Chau, Anshuman Dewangan, Margot Wagner 
# and Jin-Long Huang
# Winter 2021
################################################################################
from comet_ml import Experiment
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import torch.optim as optim
import sys

from datetime import datetime
from constants import *
from dataset_factory import getDataloaders
from model_factory import getModel
from file_utils import *
from caption_utils import * 


class _Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')

        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.name = config_data['experiment_name']

        dataset_config = config_data['dataset']
        batch_size = dataset_config['batch_size']
        num_workers = dataset_config['num_workers']
        data_file = dataset_config['data_file_path']

        experiment_config = config_data['experiment']
        self.epochs = experiment_config['num_epochs']
        learning_rate = experiment_config['learning_rate']

        model_config = config_data['model']
        hidden_size = model_config['hidden_size']
        embedding_size = model_config['embedding_size']
        self.is_variational = model_config['is_variational']
        
        generation_config = config_data['generation']
        max_length = generation_config['max_length']
        prediction_type = ("Stochastic", "Deterministic")[generation_config["deterministic"]]
        temperature = generation_config['temperature']

        self.experiment_dir = os.path.join(ROOT_STATS_DIR, self.name)

        if LOG_COMET:
            self.experiment = Experiment(
                api_key="CaoCCUZVjE0gXKA3cbtMKSSKL",
                project_name="image-captioning-251b",
                workspace="keshav919",
            )

        # Load Datasets
        tokenizer, self.train_loader, self.val_loader, self.test_loader = getDataloaders(
            data_file, max_length, batch_size, num_workers)
        
        # Setup Experiment
        self.current_epoch = 0
        self.training_losses = []
        self.bleu1_t = [] # Geeling: log bleu scores
        self.bleu4_t = []
        self.bleu1_v = [] # Keshav: log bleu scores
        self.bleu4_v = []
        self.val_losses = []
        self.best_loss = float('inf')
        self.best_model = None  # Save your best model in this field and use this in test method.

        if LOG_COMET:
            tags = [self.name, self.is_variational, prediction_type]
            hyper_params = {
                "Epochs": self.epochs,
                "Batch Size": batch_size,
                "Learning Rate": learning_rate,
                "Hidden Size": hidden_size,
                "Embedding Size": embedding_size,
                "Max Length": max_length,
                "Temperature": temperature
            }

            self.experiment.add_tags(tags)
            self.experiment.log_parameters(hyper_params)

        # Initialize Model
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.model = getModel(config_data, self.vocab_size)
        
        # TODO: need to add KL divergence
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.init_model()

        self.load_experiment() # Load Experiment Data if available


    def load_experiment(self):
        """
        Loads the experiment data if exists to resume training from last saved checkpoint.
        """
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        # Since we use comet, all our metrics are logged there rather than these directories. 
        # Create the dir just for test output
        os.makedirs(self.experiment_dir, exist_ok=True)


    def init_model(self):
        """
            Gets GPU ready to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion.to(self.device)
        
       
    def loss_function(self, raw_outputs, hypothesis, mu, logvar):
        ceLoss = self.criterion(raw_outputs, hypothesis)
#         print(ceLoss)
        
        if self.is_variational:
            klLoss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            klLoss /= len(raw_outputs)
            
            ceLoss += klLoss
#             print(klLoss)
        
        return ceLoss


    def run(self):
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.current_epoch = epoch

            train_loss, bleu1_scores_t, bleu4_scores_t = self.train()
            if LOG_COMET:
                self.experiment.log_metrics({'Train_Loss': train_loss}, epoch=epoch)
                self.experiment.log_metrics({'Train_Metric/BLEU-1': bleu1_scores_t}, epoch=epoch)
                self.experiment.log_metrics({'Train_Metric/BLEU-4': bleu4_scores_t}, epoch=epoch)
            
            val_loss, bleu1_scores_v, bleu4_scores_v = self.val()
            if LOG_COMET:
                self.experiment.log_metrics({'Val_Loss': val_loss}, epoch=epoch)
                self.experiment.log_metrics({'Val_Metric/BLEU-1': bleu1_scores_v}, epoch=epoch)
                self.experiment.log_metrics({'Val_Metric/BLEU-4': bleu4_scores_v}, epoch=epoch)

            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model, './saved_models/{}'.format(self.name))


    def train(self):
        self.model.train()
        training_loss = 0
        bleu1_scores = 0.0
        bleu4_scores = 0.0
        print_iter = 50
        if_counter = 0

        for i, (prem, hyp, lab) in enumerate(self.train_loader):
            self.model.zero_grad()

            prem = prem.long().to(self.device)
            hyp = hyp.long().to(self.device)
            lab = lab.to(self.device)

            # Forward pass
            preds, raw_outputs, mu, logvar = self.model(prem, hyp, lab, self.device, is_teacher_forcing_on=True)

            # Calculate loss and perform backprop
            loss = self.loss_function(raw_outputs[:,1:].permute(0, 2, 1), hyp[:,1:], mu, logvar)
            loss.backward()
            self.optimizer.step()

            # Log the training loss
            training_loss += loss.item()

            # View deterministic predictions
            if i % print_iter == 0:
                if_counter += 1
                
                # Get the sentence without the <start> and <end> and other tags
                clean_premise_text = clean_caption(prem, self.tokenizer)
                clean_preds_text = clean_caption(preds, self.tokenizer)
                clean_targets_text = clean_caption(hyp, self.tokenizer)

                # Calculate bleu scores
                b1 = calculate_bleu(bleu1, clean_preds_text, clean_targets_text)
                b4 = calculate_bleu(bleu4, clean_preds_text, clean_targets_text)
                bleu1_scores += (b1/len(clean_preds_text))
                bleu4_scores += (b4/len(clean_preds_text))

                print(self.current_epoch, i, ": ------ TRAIN ------")
                print("------ Actual Premise ------")
                print(clean_premise_text[0])
                print("------ Actual Hypothesis ------")
                print(clean_targets_text[0])
                print("------ Predicted Hypothesis ------")
                print(clean_preds_text[0])
                print()

        return training_loss/(i+1), bleu1_scores/if_counter, bleu4_scores/if_counter

    def val(self):
        self.model.eval()
        val_loss = 0
        bleu1_scores = 0.0
        bleu4_scores = 0.0
        print_iter = 50 
        if_counter = 0

        with torch.no_grad():
            for i, (prem, hyp, lab) in enumerate(self.val_loader):
                prem = prem.long().to(self.device)
                hyp = hyp.long().to(self.device)
                lab = lab.to(self.device)

                # Forward pass
                preds, raw_outputs, mu, logvar = self.model(prem, hyp, lab, self.device, is_teacher_forcing_on=True)

                # Calculate loss and perform backprop
                loss = self.loss_function(raw_outputs[:,1:].permute(0, 2, 1), hyp[:,1:], mu, logvar)

                # Log the training loss
                val_loss += loss.item()

                # View deterministic predictions
                if i % print_iter == 0:
                    if_counter += 1
                    
                    # Get the sentence without the <start> and <end> and other tags
                    clean_premise_text = clean_caption(prem, self.tokenizer)
                    clean_preds_text = clean_caption(preds, self.tokenizer)
                    clean_targets_text = clean_caption(hyp, self.tokenizer)

                    # Calculate bleu scores
                    b1 = calculate_bleu(bleu1, clean_preds_text, clean_targets_text)
                    b4 = calculate_bleu(bleu4, clean_preds_text, clean_targets_text)
                    bleu1_scores += (b1/len(clean_preds_text))
                    bleu4_scores += (b4/len(clean_preds_text))

                    print(self.current_epoch, i, ": ------ VALIDATION ------")
                    print("------ Actual Premise ------")
                    print(clean_premise_text[0])
                    print("------ Actual Hypothesis ------")
                    print(clean_targets_text[0])
                    print("------ Predicted Hypothesis ------")
                    print(clean_preds_text[0])
                    print()


        return val_loss/(i+1), bleu1_scores/if_counter, bleu4_scores/if_counter

    def test(self):
        self.model = torch.load('./saved_models/{}'.format(self.name))
        self.model.eval()
        test_loss = 0
        bleu1_scores = 0.0
        bleu4_scores = 0.0
        print_iter = 50 

        with torch.no_grad():
            for i, (prem, hyp, lab) in enumerate(self.test_loader):
                prem = prem.long().to(self.device)
                hyp = hyp.long().to(self.device)
                lab = lab.to(self.device)

                # Forward pass
                _, raw_outputs, mu, logvar = self.model(prem, hyp, lab, self.device, is_teacher_forcing_on=True)
                preds, _, _, _ = self.model(prem, hyp, lab, self.device, is_teacher_forcing_on=False)

                # Calculate loss and perform backprop
                loss = self.loss_function(raw_outputs[:,1:].permute(0, 2, 1), hyp[:,1:], mu, logvar)

                # Log the training loss
                test_loss += loss.item()

                # Get the sentence without the <start> and <end> and other tags
                clean_premise_text = clean_caption(prem, self.tokenizer)
                clean_preds_text = clean_caption(preds, self.tokenizer)
                clean_targets_text = clean_caption(hyp, self.tokenizer)

                # Calculate bleu scores
                b1 = calculate_bleu(bleu1, clean_preds_text, clean_targets_text)
                b4 = calculate_bleu(bleu4, clean_preds_text, clean_targets_text)
                bleu1_scores += (b1/len(clean_preds_text))
                bleu4_scores += (b4/len(clean_preds_text))
                
                if i % print_iter == 0:
                    print(i, ": ------ TEST ------")
                    print("------ Actual Premise ------")
                    print(clean_premise_text[0])
                    print("------ Actual Hypothesis ------")
                    print(clean_targets_text[0])
                    print("------ Predicted Hypothesis ------")
                    print(clean_preds_text[0])
                    print()

        bleu1_scores = bleu1_scores/(i+1)
        bleu4_scores = bleu4_scores/(i+1)
        test_loss = test_loss/(i+1)

        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss, bleu1_scores, bleu4_scores)
        self.log(result_str)

        if LOG_COMET:
            self.experiment.log_metrics({'Test_Loss': test_loss})
            self.experiment.log_metrics({'Test_Metric/BLEU-1': bleu1_scores})
            self.experiment.log_metrics({'Test_Metric/BLEU-4': bleu4_scores})

        return test_loss, bleu1_scores, bleu4_scores

    
    def save_model(self):
        root_model_path = os.path.join(self.experiment_dir, 'latest_model.pt')
        model_dict = self.model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, bleu1_t, bleu4_t, val_loss, bleu1_v, bleu4_v):
        self.training_losses.append(train_loss)
        self.bleu1_t.append(bleu1_t)
        self.bleu4_t.append(bleu4_t)
        self.val_losses.append(val_loss)
        self.bleu1_v.append(bleu1_v)
        self.bleu4_v.append(bleu4_v)

        self.plot_stats()

        write_to_file_in_dir(self.experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.experiment_dir, 'bleu1.txt', self.bleu1)
        write_to_file_in_dir(self.experiment_dir, 'bleu4.txt', self.bleu4)
        write_to_file_in_dir(self.experiment_dir, 'val_losses.txt', self.val_losses)

    def log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.experiment_dir, file_name, log_str)

    def log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.epochs - self.current_epoch - 1)
        train_loss = self.training_losses[self.current_epoch]
        val_loss = self.val_losses[self.current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.log(summary_str, 'epoch.log')