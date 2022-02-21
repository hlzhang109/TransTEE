import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Dataset_from_matrix(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_matrix):
        """
        Args: create a torch dataset from a tensor data_matrix with size n * p
        [treatment, features, outcome]
        """
        self.data_matrix = data_matrix
        self.num_data = data_matrix.shape[0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_matrix[idx, :]
        return (sample[0:-1], sample[-1])

class Dataset_ihdp(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path='./dataset/ihdp/ihdp_npci_1-1000.all.npy', replications=10):
        """
        Args: create a torch dataset from a tensor data_matrix with size n * p
        [treatment, features, outcome]
        """
        
        self.data_matrix = np.reshape(self.data_matrix, (-1, self.data_matrix.shape[1]))
        self.num_data = self.data_matrix.shape[0] * replications

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data_matrix[idx, :]
        return (sample[0:-1], sample[-1])

def get_iter(data_matrix, batch_size, shuffle=True):
    dataset = Dataset_from_matrix(data_matrix)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator

def get_iter_ihdp(path, batch_size, shuffle=True):
    dataset = Dataset_ihdp(path)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator

# Dataset_ihdp()