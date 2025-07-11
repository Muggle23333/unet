import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

from model import UNet1D  

# ============ 一些与train.py保持一致的实用函数 ============

def load_raw_sequences(folder_path):
    """加载原始变长序列数据（csv，第一列为波形）"""
    sequences = []
    names = []
    labels = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            data = pd.read_csv(os.path.join(folder_path, file), header=None).values
            numeric_data = pd.to_numeric(data[:, 0], errors='coerce')
            sequences.append(numeric_data[~pd.isna(numeric_data)].astype('float32'))
            names.append(file)
            # 支持推理时也有标签（如果有第二列）
            if data.shape[1] > 1:
                label_col = pd.to_numeric(data[:, 1], errors='coerce')
                labels.append(label_col[~pd.isna(label_col)].astype('float32'))
            else:
                labels.append(None)
    return sequences, names, labels

def normalize_sequences(sequences, scaler_path='scaler.pkl'):
    """推理时归一化：直接用训练时保存的scaler"""
    scaler = joblib.load(scaler_path)
    return [scaler.transform(seq.reshape(-1, 1)).flatten() for seq in sequences]

def pad_sequences(sequences, max_len=None, value=0.):
    if not max_len:
        max_len = max(len(seq) for seq in sequences)
    # pad到2的N次方（与训练时一致，假设为2^6=64）
    factor = 2 ** 6
    if max_len % factor != 0:
        max_len = ((max_len // factor) + 1) * factor
    padded = np.full((len(sequences), max_len), value, dtype='float32')
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
    return padded, max_len

class InferDataset(Dataset):
    def __init__(self, sequences):
        self.X = torch.tensor(sequences, dtype=torch.float32).unsqueeze(1)  # [B, 1, L]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

# ============ UNet1D网络定义（与train.py保持一致） ============

import torch.nn as nn


# ============ 推理与评估主流程 ============

def infer(input_folder, model_ckpt='best_unet_model.pt', scaler_path='scaler.pkl', output_folder='infer_results', threshold=0.5):
    # 1. 加载和预处理数据
    sequences, names, labels = load_raw_sequences(input_folder)
    X = normalize_sequences(sequences, scaler_path)
    X_padded, max_len = pad_sequences(X)
    dataset = InferDataset(X_padded)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    # 2. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet1D().to(device)
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()
    # 3. 推理
    os.makedirs(output_folder, exist_ok=True)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        idx = 0
        for X_batch in loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()  # [B, 1, L]
            for i in range(X_batch.size(0)):
                seq_len = len(sequences[idx])
                prob = y_pred[i, 0, :seq_len]  # 还原为原始长度
                bin_pred = (prob > threshold).astype('int')
                # 保存为csv: 第一列为波形（原始归一化前），第二列为概率，第三列为预测标签
                df = pd.DataFrame({
                    'wave': sequences[idx],
                    'prob': prob,
                    'pred': bin_pred
                })
                csv_name = os.path.splitext(names[idx])[0] + '_infer.csv'
                df.to_csv(os.path.join(output_folder, csv_name), index=False)
                idx += 1
                all_preds.append(bin_pred)
    print(f"推理完毕，结果已保存到 {output_folder}/")

    # ======= 性能评估模块：仅输出准确率 =======
    # 只在所有样本都带有标签时才评估
    if all([lab is not None for lab in labels]):
        total_correct = 0
        total_count = 0
        sample_accuracies = []
        for pred, label in zip(all_preds, labels):
            minlen = min(len(pred), len(label))
            total_correct += np.sum(pred[:minlen] == label[:minlen])
            total_count += minlen
            acc_i = np.sum(pred[:minlen] == label[:minlen]) / (minlen + 1e-9)
            sample_accuracies.append(acc_i)
        acc = total_correct / (total_count + 1e-9)
        print(f"Accuracy: {acc:.4f}")

        # ========== 找出准确率最高的100个样本及其准确率 ==========
        top100_indices = np.argsort(sample_accuracies)[::-1][::]  # 从大到小排序，取前100
        print("\nTop 100 samples by accuracy:")
        for rank, idx in enumerate(top100_indices, 1):
            print(f"{rank:3d}: Sample {names[idx]} - Accuracy: {sample_accuracies[idx]:.4f}")

        return all_preds, acc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="U-Net1D 地震事件推理程序")
    parser.add_argument('--input', type=str, default='test_data_new', help='测试集文件夹，csv输入')
    parser.add_argument('--model', type=str, default='best_unet_model.pt', help='模型权重路径')
    parser.add_argument('--scaler', type=str, default='scaler.pkl', help='归一化器路径')
    parser.add_argument('--output', type=str, default='infer_results', help='推理结果输出文件夹')
    parser.add_argument('--thresh', type=float, default=0.5, help='二值化阈值(默认0.5)')
    args = parser.parse_args()
    infer(args.input, args.model, args.scaler, args.output, args.thresh)

    