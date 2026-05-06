import torch
from torch.utils.data import Dataset
from config import *
import tiktoken
from datasets import load_dataset

def load_dataset_huggingFace(enc, num_articles=200):
    ds = load_dataset("wikimedia/wikipedia", "20231101.de", split="train")
    all_tokens =[]
    for article in ds[:num_articles]['text']:
        all_tokens += enc.encode(article)
    return all_tokens


def load_text(filepath):   # For Training with local files
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def build_tokenizer():
    enc = tiktoken.get_encoding("cl100k_base")
    return enc

class TextDataset(Dataset):
    def __init__(self): # Parameter: 'filepath' for local files Training
        self.enc = build_tokenizer()

        # text = load_text(filepath)
        text = load_dataset_huggingFace(self.enc)

        # self.data = torch.tensor(self.enc.encode(text))
        self.data = torch.tensor(text)
    
    def __len__(self):
        return (len(self.data) - CONTEXT_LEN) 
    
    def __getitem__(self, index):
        input_chunk = self.data[index:index + CONTEXT_LEN]
        target_chunk = self.data[index + 1:index + CONTEXT_LEN + 1]

        return input_chunk, target_chunk