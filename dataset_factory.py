import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split # For custom data-sets
from transformers import BertTokenizer

torch.manual_seed(0)

def getDataloaders(csv_file, config_data):

    all_data = TextDataset(csv_file, config_data['generation']['max_length'])
    num_train = int(len(all_data) * 0.7)
    num_val = int(len(all_data) * 0.2)
    num_test = int(len(all_data) * 0.1)
    
    torch.manual_seed(torch.initial_seed())
    train_dataset, val_dataset, test_dataset = random_split(all_data, (num_train, num_val, num_test))

    train_dataloader = DataLoader(dataset=train_dataset,
                      batch_size=config_data['dataset']['batch_size'],
                      shuffle=True,
                      num_workers=config_data['dataset']['num_workers'],
                      pin_memory=True) # Pin memory makes it faster to move from CPU to GPU, optional
    val_dataloader = DataLoader(dataset=val_dataset,
                      batch_size=config_data['dataset']['batch_size'],
                      shuffle=True,
                      num_workers=config_data['dataset']['num_workers'],
                      pin_memory=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                      batch_size=config_data['dataset']['batch_size'],
                      shuffle=False,
                      num_workers=config_data['dataset']['num_workers'],
                      pin_memory=True)
    return train_dataloader, val_dataloader, test_dataloader
    

class TextDataset(Dataset):
    def __init__(self, csv_file, max_length):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['lang_abv'] == 'en'] # Drop non-english rows 
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def __len__(self):
        return len(self.data)
    
    def _tokenize(self, x):
        # This will truncate or pad to self.max_length, such that the return array is self.max_length in length. 
        # [UNK] token is 100
        # start token is 101
        # [SEP] token is 102
        # [PAD] token is 0
        # See more here: https://huggingface.co/transformers/model_doc/bert.html?highlight=berttokenizer#berttokenizer 
        return self.tokenizer(
                x,
                max_length=self.max_length, 
                truncation=True, 
                padding='max_length')
    
    def __getitem__(self, idx):
        
        prem = self.data.iloc[idx]['premise']
        hypo = self.data.iloc[idx]['hypothesis']
        
        # Tokenize with BERT 
        # TODO: make use of 'token_type_ids' and 'attention_mask' (attention will disregard the PAD tokens)
        # Currently only uses 'input_ids' which is a tokenized representation similar to 
        premise = self._tokenize(prem)['input_ids']
        hypothesis = self._tokenize(hypo)['input_ids']
        
        return torch.Tensor(premise), torch.Tensor(hypothesis), self.data.iloc[idx]['label']