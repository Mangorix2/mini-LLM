import torch
from torch.utils.data import Dataset
from config import *
import tiktoken

def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def build_tokenizer():
    enc = tiktoken.get_encoding("cl100k_base")
    return enc

class TextDataset(Dataset):
    def __init__(self, filepath):
        text = load_text(filepath)
        self.enc = build_tokenizer()
        self.data = torch.tensor(self.enc.encode(text))
    
    def __len__(self):
        return (len(self.data) - CONTEXT_LEN) 
    
    def __getitem__(self, index):
        input_chunk = self.data[index:index + CONTEXT_LEN]
        target_chunk = self.data[index + 1:index + CONTEXT_LEN + 1]

        return input_chunk, target_chunk