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
def get_model(config_data):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']

    return VAE(embedding_size, hidden_size, vocab_size, prediction_type, temperature)

    
class VAE(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size):
        """
        Initialize pretrained ResNet50 model
        
        :param embedding_size: size of feature vector output
        """
        super(VAE, self).__init__()
        
        # Save parameters
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embedding_size)

        # TODO: Define Encoder
        self.enc_lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)


        # TODO: Define Decoder
        self.dec_lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        

        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """
        Initialize weights
        """
        # Initialize encoder layer weights

        # Initialize decoder layer weights
        torch.nn.init.xavier_uniform_(self.decoder_ll.weight)
        torch.nn.init.xavier_uniform_(self.decoder_ll.bias.reshape((-1,1)))
        
        
    def forward(self, premises, hypothesis, labels, device, is_teacher_forcing_on=True): # captions: batch_size x length 
        # TODO: Encoder
        
        pred = torch.unsqueeze(hypothesis[:, 0], 1) # All the start ids 
        # TODO: Decoder
        for i in range(1, hypothesis.shape[1]):
            embedding = self.embed(pred)
            

        return outputted_words, raw_outputs