import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
from SwinUNet import SwinUNet
from dataset import train_dataloader ,valid_dataloader


def train_epoch(model, dataloader):
    model.train()
    losses= []
    for x, y,_,_ in dataloader:
        optimizer.zero_grad()
        out = model.forward(x.to(DEVICE))
        loss = loss_fn(out, y.to(DEVICE)).to(DEVICE)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def train(model, epochs,  save_path_template, train_dataloader, valid_dataloader):

    for ep in range(epochs):
        train_loss = train_epoch(model, train_dataloader)
        
        # Compute validation loss
        valid_loss = evaluate(model, valid_dataloader)
        
        # Print train and validation loss after every 10 epochs
        if (ep + 1) % 10 == 0:
            print(f'Epoch: {ep+1}: train_loss={train_loss:.5f}, valid_loss={valid_loss:.5f}')
        
        # Save the model after every 10 epochs
        if (ep + 1) % 10 == 0:
            save_path = save_path_template.format(epoch=ep+1)
            torch.save(model.state_dict(), save_path)
        
    return train_loss

def evaluate(model, dataloader):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y,_,_ in dataloader:
            out = model.forward(x.to(DEVICE))
            loss = loss_fn(out, y.to(DEVICE)).to(DEVICE)
            losses.append(loss.item())
    return np.mean(losses)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinUNet(224,224,30,32,1,3,4).to(DEVICE)

for p in model.parameters():
    if p.dim() > 1:
            nn.init.kaiming_uniform_(p)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
save_path_template = 'Model_weights/model_epoch_{epoch}.pth'
train(model, epochs=100, save_path_template=save_path_template,train_dataloader=train_dataloader, valid_dataloader=valid_dataloader)

