import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SmallLM
from data import TextDataset
from config import *
import time
import glob
import os

def save_checkpoint(model, step, epoch=False):
    checkpoints = sorted(
        glob.glob("checkpoint/model_step*.pt"),
        key=lambda x: int(x.split('step')[1].split('.')[0])
    )
    while len(checkpoints) > 4:
        os.remove(checkpoints[0])
        checkpoints.pop(0)

    if(epoch):
        torch.save(model.state_dict(), f"epochs/epoch{step}.pt")
        return
    
    torch.save(model.state_dict(), f"checkpoint/model_step{step}.pt")

    

def train(resume_from=None, start_step=0):

    global_step = start_step

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training auf: {device}")

    dataset = TextDataset('data/input.txt')
    dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SmallLM().to(device)
    if resume_from:
        model.load_state_dict(torch.load(resume_from))
        print(f"Model geladen: {resume_from}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    NUM_EPOCHS = 1

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        for step, (input_batch, target_batch) in enumerate(dataLoader):
            global_step += 1

            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            logits = model(input_batch)
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), target_batch.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            if global_step % 100 == 0:
                print(f"Epoch {epoch + 1}, Step {global_step}, Loss: {loss.item():.4f}")

            if global_step % 500 == 0:
                save_checkpoint(model, global_step)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1 } fertig – Zeit: {epoch_time:.1f}s")
        save_checkpoint(model, epoch + 1, epoch=True)


if __name__ == '__main__':
    train(resume_from='checkpoint/model_step12000.pt', start_step=12000)