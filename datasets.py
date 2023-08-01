## Dataloader
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from torch.utils.data import DataLoader

class simple_traj_dataset(Dataset):
    
    def __init__(self, data_path):
        
        df = pd.read_csv(data_path)
        self.df = df.iloc[:, 0:6]
        self.index_list = list(self.df['algo'].unique())


    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):

        df_trajectory = self.df.query(f"algo == {self.index_list[idx]}")

        trajectory = np.transpose(df_trajectory[['x', 'y']].values)

        return torch.Tensor(trajectory)

class chess_dataset(Dataset):
    
    def __init__(self, data_path):
        
        df = pd.read_csv(data_path)
        self.df = df.iloc[:, 0:6]
        self.index_list = list(self.df['line'].unique())


    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):

        df_trajectory = self.df.query(f"algo == {self.index_list[idx]}")

        trajectory = np.transpose(df_trajectory[['x', 'y']].values)

        return torch.Tensor(trajectory)


class mast_dataset(Dataset):
    
    def __init__(self, data_path):
        
        self.df = pd.read_csv(data_path)
        #self.df = df.iloc[:, 0:6]
        self.index_list = list(self.df['sample_idx'].unique())


    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):

        df_trajectory = self.df.query(f"sample_idx == {self.index_list[idx]}")

        trajectory = np.transpose(df_trajectory[['X', 'Y']].values)

        return torch.Tensor(trajectory)
    
    #def collate_fn(self, data):
    #    print(type(data))
    #    print(len(data))
    #    print(data)


class cluster_dataset(Dataset):
    
    def __init__(self, data, labels):
        
        self.data = data
        self.labels = labels


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        data_item = self.data[idx, :,:]
        label_item = self.labels[idx]
        

        return torch.Tensor(data_item), label_item