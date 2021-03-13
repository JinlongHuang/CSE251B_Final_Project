import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split # For custom data-sets

# TODO: Remove this after we finish debugging!
torch.manual_seed(0)

def getDataloaders(csv_file_paths, max_length, batch_size, num_workers, tokenizer, val_split=0.1, test_split=0.1, class_label=-1):
    
    # Concatenate all TextDatasets into CSV filepath
    train_datasets = []
    val_datasets = []
    test_datasets = []
    for csv_file in csv_file_paths: 
        dataset = TextDataset(csv_file, max_length, tokenizer, class_label=class_label)
        num_val = int(len(dataset) * val_split)
        num_test = int(len(dataset) * test_split)
        num_train = int(len(dataset) - num_val - num_test)
        
        torch.manual_seed(torch.initial_seed())
        train_dataset, val_dataset, test_dataset = random_split(dataset, (num_train, num_val, num_test))
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)
        test_datasets.append(test_dataset)

    # Concatenate our datasets
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    test_dataset = torch.utils.data.ConcatDataset(val_datasets)

    # Create dataloaders
    train_dataloader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          pin_memory=True) # Pin memory makes it faster to move from CPU to GPU, optional
    val_dataloader   = DataLoader(dataset=val_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          pin_memory=True)
    test_dataloader  = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True)
    return train_dataloader, val_dataloader, test_dataloader
    

class TextDataset(Dataset):
    def __init__(self, csv_file, max_length, tokenizer, class_label=-1):
        self.data = pd.read_csv(csv_file)
        
        self.translated = (False, True)['translated' in csv_file]

        if not self.translated:
            self.data = self.data[self.data['lang_abv'] == 'en'] # Drop non-english rows 
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.csv_file = csv_file
        
        if class_label == 0:
            self.data = self.data[self.data['label'] == 0]
        elif class_label == 1:
            self.data = self.data[self.data['label'] == 1]
        elif class_label == 2:
            self.data = self.data[self.data['label'] == 2]
        
    def __len__(self):
        print(len(self.data))
        return len(self.data)
    
    def _tokenize(self, x):
        # This will truncate or pad to self.max_length, such that the return array is self.max_length in length. 
        # [UNK] token is 100
        # start token is 101
        # [SEP] token is 102
        # [PAD] token is 0
        # See more here: https://huggingface.co/transformers/model_doc/bert.html?highlight=berttokenizer#berttokenizer 
        
        try: 
            toReturn = self.tokenizer(
                str(x), 
                max_length=self.max_length, 
                truncation=True, 
                padding='max_length')
        except Exception as e: 
            # TODO: (low pri) There's a nan in the snli dataset but I can't find it in the dataframe that gets saved as csv XD
            # For now, str(x) will turn NaN into a string and pass it back like that. 
            print(self.csv_file)
            print(x) 
            print(type(x))
            print(e) 
            raise e
        return toReturn
    

    def __getitem__(self, idx):
        if self.translated:
            prem = self.data.iloc[idx]['premise_translated']
            hypo = self.data.iloc[idx]['hypothesis_translated']
        else:
            prem = self.data.iloc[idx]['premise']
            hypo = self.data.iloc[idx]['hypothesis']
        
        # Tokenize with BERT 
        # Make use of 'token_type_ids' and 'attention_mask' (attention will disregard the PAD tokens)

        prem_id = self._tokenize(prem)['input_ids']
        hypo_id = self._tokenize(hypo)['input_ids']

        prem_att_mask = self._tokenize(prem)['attention_mask']
        hypo_att_mask = self._tokenize(hypo)['attention_mask']
        
        return torch.Tensor(prem_id), torch.Tensor(hypo_id), torch.Tensor(prem_att_mask), torch.Tensor(hypo_att_mask), self.data.iloc[idx]['label']