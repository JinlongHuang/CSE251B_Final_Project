################################################################################
# CSE 251 B: Final Project
# Code by Keshav Rungta, Geeling Chau, Anshuman Dewangan, Margot Wagner 
# and Jin-Long Huang
# Winter 2021
################################################################################
import torch.nn as nn
import torch
import sys
from caption_utils import *
from transformers import BertForSequenceClassification

# Build and return the model here based on the configuration.
def getModel(config, vocab_size):
    embedding_size = config['model']['embedding_size']
    hidden_size = config['model']['hidden_size']
    is_variational = config['model']['is_variational']
    is_vae = config['model']['is_vae']
    
    deterministic = config['generation']['deterministic']
    temperature = config['generation']['temperature']
    max_length = config['generation']['max_length']
    
    if is_vae:
        return VAE(embedding_size, hidden_size, vocab_size, deterministic, temperature, max_length, is_variational)
    else:
        return BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    
class VAE(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, deterministic, temperature, max_length, is_variational):
        """
            Variational Autoencoder 
        """
        super(VAE, self).__init__()

        # Save parameters
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.is_variational = is_variational
        
        self.deterministic = deterministic
        self.temperature = temperature

        self.vocab_size = vocab_size
        
        # Embedding layer to transform prem and hypo to embedding size 
        self.embed = nn.Embedding(vocab_size, embedding_size) # Also used for decoder
        self.enc_lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)

        # Reparameterization layer
        self.mu_ll = nn.Linear(hidden_size, hidden_size)
        self.logvar_ll = nn.Linear(hidden_size, hidden_size)

        # Define Decoder
        self.dec_lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder_ll = nn.Linear(hidden_size, vocab_size)

        # Initialize weights
        self.init_weights()
        
        
    def init_weights(self):
        """
        Initialize weights
        """
        # Initialize encoder layer weights
        torch.nn.init.xavier_uniform_(self.mu_ll.weight)
        torch.nn.init.xavier_uniform_(self.mu_ll.bias.reshape((-1,1)))
        torch.nn.init.xavier_uniform_(self.logvar_ll.weight)
        torch.nn.init.xavier_uniform_(self.logvar_ll.bias.reshape((-1,1)))

        # Initialize decoder layer weights
        torch.nn.init.xavier_uniform_(self.decoder_ll.weight)
        torch.nn.init.xavier_uniform_(self.decoder_ll.bias.reshape((-1,1)))

        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
        
    def forward(self, premises, hypothesis, labels, device, is_teacher_forcing_on=True):
        # Replace start tag with the label; batch x max_len
        # Anshuman: I'm unsure about this. Our model should work without it
#         premises[:, 0] = labels 
        
        # Encode premise features 
        prem_embedded = self.embed(premises) # batch x max_len x embedding_size

        _, hidden = self.enc_lstm(prem_embedded) # hidden is the set of feats that will be passed to decoder
        
        # Initialize variables
        mu0 = 0
        log_var0 = 0

        if self.is_variational:
            # Sample using reparameterization
            ## Hidden state
            mu0 = self.mu_ll(hidden[0].permute(1,0,2))
            log_var0 = self.logvar_ll(hidden[0].permute(1,0,2))
            z0 = self.reparameterize(mu0, log_var0).permute(1,0,2) # 1 x batch x hidden_size

            # Anshuman: Divyanshu suggests we don't need this
            ## Cell state
#             mu1 = self.mu_ll(hidden[1].permute(1,0,2))
#             log_var1 = self.logvar_ll(hidden[1].permute(1,0,2))
#             z1 = self.reparameterize(mu1, log_var1).permute(1,0,2) # 1 x batch x hidden_size

            hidden = (z0, z0)

        # Decoder
        outputted_words = torch.zeros(hypothesis.shape).to(device) # batch x max_len
        raw_outputs = torch.zeros((hypothesis.shape[0], hypothesis.shape[1], self.vocab_size)).to(device) # batch x max_len x vocab_size

        outputted_words[:, 0] = hypothesis[:, 0] # Initialize the output with start id
        pred = torch.unsqueeze(hypothesis[:, 0], 1) # All the start ids; batch x 1

        for i in range(1, hypothesis.shape[1]):
            embedding = self.embed(pred) # batch x 1 x embedding_size

            # Run through LSTM
            # lstm_out: batch x 1 x hidden_size
            # hidden:   1 x batch x hidden_size
            lstm_out, _ = self.dec_lstm(embedding, hidden) # Anshuman: Divyanshu suggests passing initial hidden state every time

            # Create raw output
            outputs = self.decoder_ll(lstm_out) # batch x 1 x vocab_size

            # Save raw result
            raw_outputs[:, i, :] = outputs.squeeze() # batch x max_len x vocab_size

            # Get predicted word 
            if self.deterministic:
                pred = torch.argmax(outputs, dim=2) # batch x 1
            else:
                pred = stochastic_generation(outputs, self.temperature)

            # Save the word result 
            outputted_words[:, i] = pred.squeeze() # batch x max_len

            # If we're training, use teacher forcing instead
            if is_teacher_forcing_on:
                pred = torch.unsqueeze(hypothesis[:, i],1) # batch x 1
        
        return outputted_words, raw_outputs, mu0, log_var0