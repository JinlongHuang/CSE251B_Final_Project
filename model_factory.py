################################################################################
# CSE 251 B: Final Project
# Code by Keshav Rungta, Geeling Chau, Anshuman Dewangan, Margot Wagner 
# and Jin-Long Huang
# Winter 2021
################################################################################
import torch.nn as nn
import torch
import torchvision
import sys
from caption_utils import *

# Build and return the model here based on the configuration.
def getModel(config, vocab_size):
    embedding_size = config['model']['embedding_size']
    hidden_size = config['model']['hidden_size']
    deterministic = config['generation']['deterministic']
    temperature = config['generation']['temperature']
    
    return VAE(embedding_size, hidden_size, vocab_size, deterministic, temperature)

    
class VAE(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, deterministic, temperature):
        """
            Variational Autoencoder 
            TODO: Test that this works and make this variational  LOL

        """
        super(VAE, self).__init__()

        # Save parameters
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        
        self.deterministic = deterministic
        self.temperature = temperature

        self.vocab_size = vocab_size
        
        # Embedding layer to transform prem and hypo to embedding size 
        self.embed = nn.Embedding(vocab_size, embedding_size) # Also used for decoder
        self.enc_lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)

        # Define Decoder
        self.dec_lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder_ll = nn.Linear(hidden_size, vocab_size) # linear layer

        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """
        Initialize weights
        """
        # Initialize encoder layer weights
        # None :D 

        # Initialize decoder layer weights
        torch.nn.init.xavier_uniform_(self.decoder_ll.weight)
        torch.nn.init.xavier_uniform_(self.decoder_ll.bias.reshape((-1,1)))
        
    def forward(self, premises, hypothesis, labels, device, is_teacher_forcing_on=True):

        # Encode premise features
        premises[:, 0] = labels # Replace start tag with the label
        prem_embedded = self.embed(premises)

        batch_size = premises.shape[0]
        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        enc_hidden = (h_0,h_0)
        outputs, hidden = self.enc_lstm(prem_embedded, enc_hidden) # hidden is the set of feats that will be passed to decoder

        # Decoder
        outputted_words = torch.zeros(hypothesis.shape).to(device) # batch_size x 20 length
        raw_outputs = torch.zeros((hypothesis.shape[0], hypothesis.shape[1], self.vocab_size)).to(device) # batch_size x 20 x vocab_size

        outputted_words[:, 0] = hypothesis[:, 0] # Initialize the output with start id
        pred = torch.unsqueeze(hypothesis[:, 0], 1) # All the start ids 

        for i in range(1, hypothesis.shape[1]):
            embedding = self.embed(pred)

            # Run through LSTM
            lstm_out, hidden = self.dec_lstm(embedding, hidden)

            # Create raw output
            outputs = self.decoder_ll(lstm_out) 

            # Save raw result
            raw_outputs[:, i, :] = outputs.squeeze()

            # Get predicted word 
            if self.deterministic:
                pred = torch.argmax(outputs, dim=2) # 64 x 1
            else:
                pred = stochastic_generation(outputs, self.temperature)

            # Save the word result 
            outputted_words[:, i] = pred.squeeze() # 64

            # If we're training, use teacher forcing instead
            if is_teacher_forcing_on:
                pred = torch.unsqueeze(hypothesis[:, i],1) # 64 x 1
        
        return outputted_words, raw_outputs