import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


import cv2
import time 
from sklearn.model_selection import train_test_split


from config import cfg
from dataset import SCDataset
from engine import train_epoch, valid_epoch
from model import SkinCancerModel
from transform import get_transforms
from lr_scheduler import get_linear_schedule_with_warmup



def main():
    
    device = torch.device('cuda')

    data = pd.read_csv('/skin-cancer-data/data.csv')
            
    data['label'] = data['label'].map(cfg.label_dict)
    
    train_transform, valid_transform = get_transforms()
    
    #Spliting data for trainning and validation 
    train, valid = train_test_split(data, test_size = 0.25, random_state = cfg.seed, stratify = data['label'])
    train, valid = train.reset_index(), valid.reset_index()
    
    #Create Dataset and DataLoader
    train_dataset = SCDataset(train, transform = train_transform)
    valid_dataset = SCDataset(valid, transform = valid_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.batch_size, num_workers=4)
    
    #Create Model
    model = SkinCancerModel().to(device)
    
    #Loss function for our trainning
    criterion = nn.CrossEntropyLoss()
    
    #Optimizer for updating weights
    optimizer = optim.Adam(model.parameters(), lr = 5e-5)
    
    #Learning Rate Scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, len(train_loader), cfg.epochs * len(train_loader))
    
    best_loss = 999.99
    for epoch in range(cfg.epochs):
        
        start_time = time.time()
        
        train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        loss = valid_epoch(model, train_loader, criterion, device)
        
        if loss < best_loss:
            torch.save(model.state_dict(), f'{cfg.model_name}_{epoch}.pth')
            
        time_taken = time.time() - start_time
        
        
        print('Epoch {:2d} | val_Loss: {:.4f}  | {:d}s'.format(epoch, loss, int(time_taken)))