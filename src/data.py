import torch
from torch.utils.data import Dataset
from config import *

def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def build_tokenizer(text):
    characters = sorted(set(text))

    char_to_int = {}
    for index, char in enumerate(characters):
        char_to_int[char] = index

    int_to_char = {}
    for index, char in enumerate(characters):
        int_to_char[index] = char

    return char_to_int, int_to_char

class TextDataset(Dataset):
    def __init__(self, filepath):
        text = load_text(filepath)
        self.char_to_int, self.int_to_char = build_tokenizer(text)
        self.data = torch.tensor([self.char_to_int[c] for c in text])
    
    def __len__(self):
        return (len(self.data) - CONTEXT_LEN) 
    
    def __getitem__(self, index):
        input_chunk = self.data[index:index + CONTEXT_LEN]
        target_chunk = self.data[index + 1:index + CONTEXT_LEN + 1]

        return input_chunk, target_chunk