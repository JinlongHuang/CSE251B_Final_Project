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

# Build and return the model here based on the configuration.
def getModel(hidden_size, embedding_size, deterministic, temperature, vocab_size):

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
        self.embed = nn.Embedding(vocab_size, embedding_size) 

        # Define Encoder 
        self.encoder_ll = nn.Linear(embedding_size, hidden_size)

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
        torch.nn.init.xavier_uniform_(self.encoder_ll.weight)
        torch.nn.init.xavier_uniform_(self.encoder_ll.bias.reshape((-1,1)))

        # Initialize decoder layer weights
        torch.nn.init.xavier_uniform_(self.decoder_ll.weight)
        torch.nn.init.xavier_uniform_(self.decoder_ll.bias.reshape((-1,1)))
        
        
    def forward(self, premises, hypothesis, labels, device, is_teacher_forcing_on=True):
        # Encode premise features
        prem_embedded = self.embed(premises)

        feat = self.encoder_ll(prem_embedded)
        feat = torch.unsqueeze(feat,0) # Reshape for format that lstm is happy with
        # Decoder
        outputted_words = torch.empty(hypothesis.shape).to(device) # batch_size x 20 length
        raw_outputs = torch.empty((hypothesis.shape[0], hypothesis.shape[1], self.vocab_size)).to(device) # batch_size x 20 x vocab_size

        outputted_words[:, 0] = hypothesis[:, 0] # Initialize the output with start id
        pred = torch.unsqueeze(hypothesis[:, 0], 1) # All the start ids 

        hidden = (feat, feat)
        for i in range(1, hypothesis.shape[1]):
            embedding = self.embed(pred)

            # Run through LSTM
            ltsm_out, hidden = self.dec_lstm(embedding, hidden)

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