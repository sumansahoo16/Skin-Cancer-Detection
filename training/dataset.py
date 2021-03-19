import os
import torch
from torch.utils.data import Dataset


class SCDataset(Dataset):
    def __init__(self, csv, transform):
        self.csv = csv
        self.transform = transform
        
    def __len__(self):
        return self.csv.shape[0]
    
    def __getitem__(self, idx):
        
        image = cv2.imread(self.csv['Id'][idx] + '.jpg')
        try : image.shape
        except : image = cv2.imread('ISIC_0027419' + '.jpg') 
        #image = self.transform(image=image)['image']
        image = image / 255
        
        image = image.transpose(2, 0, 1)
        
        return torch.tensor(image).float(), torch.tensor(self.csv['label'][idx]).long()