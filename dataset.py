import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import os
import torch.nn.functional


class SoilDataset(Dataset):
    def __init__(self, directory, split='train', sequence_length=30, transform=None, nan_replacement=0):
        self.sequence_length = sequence_length
        self.file_paths = sorted([os.path.join(directory, split, f) for f in os.listdir(os.path.join(directory, split)) if f.endswith('.tif')],
                                 key=lambda x: datetime.strptime(os.path.basename(x).split('_')[1].split('.')[0], '%Y-%m-%d'))
        self.data, self.min_max = self._load_data()
        self.transform = transform

    def _load_data(self):
        data = []
        min_max = []
        for file_path in self.file_paths:
            date_str = os.path.basename(file_path).split('_')[1].split('.')[0]
            date = datetime.strptime(date_str, '%Y-%m-%d')
            with rasterio.open(file_path) as src:
                img = src.read(1).astype(np.float32)
                
                # Handle NaN values
                img[np.isnan(img)] = 0.0 
                
                data.append(img)
                min_val, max_val = img.min(), img.max()
                min_max.append((min_val, max_val))
        return data, min_max
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, index):
        x_start = index
        x_end = index + self.sequence_length
        
        # Input sequence
        x = [self.data[i] for i in range(x_start, x_end)]
        
        # Target sequence
        y_index = x_end
        y = self.data[y_index]
        
        # Normalize input sequence
        # for i in range(len(x)):
        #     min_val, max_val = self.min_max[x_start + i]
        #     x[i] = (x[i] - min_val) / (max_val - min_val)
        
        # # Normalize target image
        # min_val, max_val = self.min_max[y_index]
        # y = (y - min_val) / (max_val - min_val)

        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float().unsqueeze(0)  # Adding channel 
        input_file_paths = [os.path.basename(self.file_paths[i]).split('_')[1].split('.')[0] for i in range(x_start, x_end)]
       
        target_file_path = os.path.basename(self.file_paths[y_index]).split('_')[1].split('.')[0]  # Include file path for both input and target
        
        return x, y,input_file_paths,target_file_path

# Directory containing train, valid, and test folders
directory = 'Dataset'

# Define datasets and data loaders for train, valid, and test sets
train_dataset = SoilDataset(directory, split='train_new',nan_replacement=0)
valid_dataset = SoilDataset(directory, split='valid_new')
test_dataset = SoilDataset(directory, split='test_new')

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Testing the dataloaders
for x, y,_,_ in train_dataloader:
    print("Train:", x.shape, y.shape)
    break

for x, y,_,_  in valid_dataloader:
    print("Valid:", x.shape, y.shape)
    break

for x, y,_,_  in test_dataloader:
    print("Test:", x.shape, y.shape)
    break