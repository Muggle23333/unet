import os
import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, random_split


def load_raw_sequences(folder_path):
    """加载原始变长序列数据（csv，第一列为波形，第二列为标签）"""
    sequences = []
    labels = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            data = pd.read_csv(os.path.join(folder_path, file), header=None).values
            numeric_data = pd.to_numeric(data[:, 0], errors='coerce')
            sequences.append(numeric_data[~pd.isna(numeric_data)].astype('float32'))
            numeric_data = pd.to_numeric(data[:, 1], errors='coerce')
            labels.append(numeric_data[~pd.isna(numeric_data)].astype('float32'))
    return sequences, labels

def normalize_sequences(sequences, mode='train'):
    """归一化变长序列"""
    if mode == 'train':
        all_values = np.concatenate(sequences)
        scaler = StandardScaler().fit(all_values.reshape(-1, 1))
        joblib.dump(scaler, 'scaler.pkl')
    else:
        scaler = joblib.load('scaler.pkl')
    return [scaler.transform(seq.reshape(-1, 1)).flatten() for seq in sequences]

def pad_sequences(sequences, max_len=None, value=0.):
    if not max_len:
        max_len = max(len(seq) for seq in sequences)
    # pad到2的N次方
    factor = 2 ** 6  # 6层pool
    if max_len % factor != 0:
        max_len = ((max_len // factor) + 1) * factor
    padded = np.full((len(sequences), max_len), value, dtype='float32')
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    return padded

class SeismicDataset(Dataset):
    """将地震序列数据封装为 PyTorch 能使用的 标准数据集格式"""
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.float32).unsqueeze(1) # [B, 1, L]
        self.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)    # [B, 1, L]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    