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


def interaktive_generator(model, tokenizer, start_txt, top_n=5):
    model.eval()

    input_indices = tokenizer.encode(start_txt)
    input_tensor = torch.tensor(input_indices).unsqueeze(0)

    while True:
        input_cropped = input_tensor[:, -CONTEXT_LEN:]
        logits = model(input_cropped)
        probs = torch.softmax(logits[0,-1,:], dim=-1)

        values, indices = torch.topk(probs, 5)

        current_text = tokenizer.decode(input_cropped[0].tolist())
        print(f"\nText: {current_text}")
        for i, (val, idx) in enumerate(zip(values, indices)):
            wort = tokenizer.decode([idx])
            print(f"  {i}: {wort!r}  ({val:.2%})")

        choice = input("Wahl (q=quit): ")
        if choice == 'q':
            break
        
        chosen_token = indices[int(choice)]
        input_tensor = torch.cat([input_tensor, chosen_token.unsqueeze(0)], dim=-1)


if __name__ == '__main__':
    model = SmaLLM()
    model.load_state_dict(torch.load('epochs/epoch4.pt'))
    
    tokenizer = build_tokenizer()
    
    output = interaktive_generator(model, tokenizer, start_text="To Be")
    print(output)