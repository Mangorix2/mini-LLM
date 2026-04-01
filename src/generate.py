import torch
from model import SmaLLM
from data import build_tokenizer, load_text
from config import *

def generator(model, tokenizer, start_text, max_new_tokes=200):
    model.eval() # bringt nichts im Moment aber später bei stärkeren LLMs schon

    input_indices = tokenizer.encode(start_text)
    input_tensor = torch.tensor(input_indices).unsqueeze(0)

    for i in range(max_new_tokes):
        input_cropped = input_tensor[:, -CONTEXT_LEN:]

        logits = model(input_cropped)
        next_token_logits = logits[0,-1,:]

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=-1)  
    
    output_indices = input_tensor[0].tolist()
    return tokenizer.decode(output_indices)

if __name__ == '__main__':
    model = SmaLLM()
    model.load_state_dict(torch.load('epochs/epoch4.pt'))
    
    tokenizer = build_tokenizer()
    
    output = generator(model, tokenizer, start_text="To Be")
    print(output)