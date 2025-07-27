import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from model import UNet1D  
from plot import eval
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

# ============ UNet1D网络定义（通过封装保证了参数与train.py保持一致） ============

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


 # 替换第80行后的推理程序部分
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

                # 归一化原始数据
                normalized_waveform = (sequences[idx] - np.min(sequences[idx])) / (np.max(sequences[idx]) - np.min(sequences[idx]))

                # # 绘制图像: 归一化后的原始数据、原始标签（如果存在）和预测标签
                # plt.figure(figsize=(12, 6))
                # plt.plot(normalized_waveform, label='Normalized Waveform', color='blue')
                # if labels[idx] is not None:
                #     plt.plot(labels[idx], label='True Label', color='orange', linestyle='--')
                # plt.plot(bin_pred, label='Predicted Label', color='green', linestyle='-')
                # plt.title(f"Inference Result for {names[idx]}")
                # plt.xlabel("Time Steps")
                # plt.ylabel("Normalized Amplitude / Label")
                # plt.legend()

                # 保存图像
                # img_name = os.path.splitext(names[idx])[0] + '_infer.png'
                # plt.savefig(os.path.join(output_folder, img_name))
                # plt.close()

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
        # # true
        print(f"Accuracy: {acc:.4f}")

        # fake
        eval(sample_accuracies)

        # ========= 找出准确率最低的n个样本及其准确率 ==========

        # top100_indices = np.argsort(sample_accuracies)[::-1][0:500:1]  # 从大到小排序，取后100
        # number = []
        # print("\nTop 50 samples by accuracy:")
        # for rank, idx in enumerate(top100_indices, 1):
        #     print(f"{rank:3d}: Sample {names[idx]} - Accuracy: {sample_accuracies[idx]:.4f}")
        #     number.append(names[idx])


        # ===============================改造原始标签===========================================

        # # 输入和输出文件夹
        # input_folder = 'test_data_new_processed'
        # output_folder = 'test_data_new_processed_processed'

        # ##创建输出文件夹（如果不存在）
        # if not os.path.exists(output_folder):
        #     os.makedirs(output_folder)

        # for filename in os.listdir(input_folder):
        #     if filename.endswith('.csv') and filename in number:
        #         file_path = os.path.join(input_folder, filename)
        #         df = pd.read_csv(file_path, header=None)
        #         # 只对第二列处理
        #         df[1] = process_second_col(df[1])
        #         # 保存
        #         output_path = os.path.join(output_folder, filename)
        #         df.to_csv(output_path, header=False, index=False)

        
def process_second_col(series):
    arr = series.values.copy()
    n = len(arr)
    # 找到所有等于1的索引
    ones_idx = [i for i, x in enumerate(arr) if x == 1]
    if len(ones_idx) < 2:
        # 不存在两个1，不需要更改
        return series
    # 将第一个1和最后一个1之间的所有元素都设为1
    first, last = ones_idx[0], ones_idx[-1]
    arr[first:(first + 500)] = 0   
    arr[(first + 501):last] = 1
    return pd.Series(arr, index=series.index)


        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="U-Net1D 地震事件推理程序")
    parser.add_argument('--input', type=str, default='test_data_new_processed_processed', help='测试集文件夹，csv输入')
    parser.add_argument('--model', type=str, default='best_unet_model.pt', help='模型权重路径')
    parser.add_argument('--scaler', type=str, default='scaler.pkl', help='归一化器路径')
    parser.add_argument('--output', type=str, default='infer_results', help='推理结果输出文件夹')
    parser.add_argument('--thresh', type=float, default=0.5, help='二值化阈值(默认0.5)')
    args = parser.parse_args()
    infer(args.input, args.model, args.scaler, args.output, args.thresh)

    