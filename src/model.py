import torch
import torch.nn as nn
from config import *

class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.query = nn.Linear(EMBEDDING_DIM, HEAD_DIM, bias=False)
        self.key   = nn.Linear(EMBEDDING_DIM, HEAD_DIM, bias=False)
        self.value = nn.Linear(EMBEDDING_DIM, HEAD_DIM, bias=False)

    def forward(self, input_tensor):
        # Inputtensor Muss (batch_size, sequence_length, EMBEDDING_DIM) sein

        queries =  self.query(input_tensor)
        keys = self.key(input_tensor)
        values = self.value(input_tensor)

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores / (HEAD_DIM ** 0.5 )

        mask = torch.tril(torch.ones(input_tensor.shape[1], input_tensor.shape[1]))
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = attention_weights @ values
        return output
        