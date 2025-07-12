import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import Counter
import joblib
from matplotlib.gridspec import GridSpec
import random
from sklearn.model_selection import train_test_split


from dataset import load_raw_sequences, normalize_sequences, pad_sequences,SeismicDataset
from plot import plot_dataset_split
from model import UNet1D

plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =================== 训练与评估 ===================

def plot_training_curves(train_log, val_log, metric_names):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for i, metric in enumerate(metric_names):
        ax = axes[i//2, i%2]
        ax.plot(train_log[metric], label=f'训练{metric}')
        ax.plot(val_log[metric], label=f'验证{metric}')
        ax.set_title(f'{metric}变化曲线')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.show()

def evaluate_model(model, dataloader, device):
    model.eval()
    loss_fn = nn.BCELoss()
    total_loss, total_acc, total_prec, total_rec, n = 0, 0, 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            y_pred_bin = (y_pred > 0.5).float()
            total_loss += loss.item() * X.size(0)
            total_acc += ((y_pred_bin == y).float().mean()).item() * X.size(0)
            tp = ((y_pred_bin == 1) & (y == 1)).sum().item()
            fp = ((y_pred_bin == 1) & (y == 0)).sum().item()
            fn = ((y_pred_bin == 0) & (y == 1)).sum().item()
            prec = tp / (tp + fp + 1e-7)
            rec = tp / (tp + fn + 1e-7)
            total_prec += prec * X.size(0)
            total_rec += rec * X.size(0)
            n += X.size(0)
    avg_loss = total_loss / n
    avg_acc = total_acc / n
    avg_prec = total_prec / n
    avg_rec = total_rec / n
    f1 = 2 * avg_prec * avg_rec / (avg_prec + avg_rec + 1e-7)
    print(f"loss: {avg_loss:.4f} | acc: {avg_acc:.4f} | prec: {avg_prec:.4f} | rec: {avg_rec:.4f} | f1: {f1:.4f}")
    return avg_loss, avg_acc, avg_prec, avg_rec, f1

class BCEWithRecallLoss(nn.Module):
    def __init__(self, lambda_recall=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.lambda_recall = lambda_recall

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        preds = (inputs > 0.5).float()
        tp = (preds * targets).sum()
        fn = ((1 - preds) * targets).sum()
        recall = tp / (tp + fn + 1e-8)  # 加1e-8防止除0
        loss = bce_loss + self.lambda_recall * (1 - recall)
        return loss
    

def train_model():
    # 1. 数据加载与预处理
    sequences, labels = load_raw_sequences(r'train_data_new_new')
    X = normalize_sequences(sequences)
    y = labels
    
    # 2. 按顺序划分数据集（前85%训练，后15%验证）
    total_size = len(X)
    train_size = int(total_size * 5/6)  # 计算训练集大小
    
    # 顺序划分
    X_train = X[:train_size]
    X_val = X[train_size:]
    y_train = y[:train_size]
    y_val = y[train_size:]
    
    # 3. 可视化数据集划分
    print("\n数据集划分统计:")
    print(f"总样本数: {total_size} | 训练集样本数: {len(X_train)} | 验证集样本数: {len(X_val)}")
    print(f"训练集平均长度: {np.mean([len(x) for x in X_train]):.1f}±{np.std([len(x) for x in X_train]):.1f}")
    print(f"验证集平均长度: {np.mean([len(x) for x in X_val]):.1f}±{np.std([len(x) for x in X_val]):.1f}")
    
    # 添加类别分布统计（按第一个时间步的标签）
    train_first_labels = [label[0] for label in y_train]
    val_first_labels = [label[0] for label in y_val]
    print(f"训练集首标签分布: 正样本 {sum(train_first_labels)} | 负样本 {len(train_first_labels)-sum(train_first_labels)}")
    print(f"验证集首标签分布: 正样本 {sum(val_first_labels)} | 负样本 {len(val_first_labels)-sum(val_first_labels)}")
    
    plot_dataset_split(X_train, X_val, y_train, y_val)

    # 将地震序列数据封装为 PyTorch 能使用的 标准数据集格式
    train_set = SeismicDataset(X_train, y_train)
    val_set = SeismicDataset(X_val, y_val)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    # 4. 模型与优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet1D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = BCEWithRecallLoss()
    # 5. 训练循环
    train_log = {'loss': [], 'acc': [], 'prec': [], 'rec': [], 'f1': []}
    val_log = {'loss': [], 'acc': [], 'prec': [], 'rec': [], 'f1': []}
    best_val_loss = float('inf')
    patience, wait = 1, 0
    epochs = 80
    for epoch in range(epochs):
        model.train()
        total_loss, total_acc, total_prec, total_rec, n = 0, 0, 0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            y_pred_bin = (y_pred > 0.5).float()
            total_loss += loss.item() * X.size(0)
            total_acc += ((y_pred_bin == y).float().mean()).item() * X.size(0)
            tp = ((y_pred_bin == 1) & (y == 1)).sum().item()
            fp = ((y_pred_bin == 1) & (y == 0)).sum().item()
            fn = ((y_pred_bin == 0) & (y == 1)).sum().item()
            prec = tp / (tp + fp + 1e-7)
            rec = tp / (tp + fn + 1e-7)
            total_prec += prec * X.size(0)
            total_rec += rec * X.size(0)
            n += X.size(0)
        avg_loss = total_loss / n
        avg_acc = (total_acc / n)  
        avg_prec = total_prec / n
        avg_rec = total_rec / n
        f1 = 2 * avg_prec * avg_rec / (avg_prec + avg_rec + 1e-7)
        train_log['loss'].append(avg_loss)
        train_log['acc'].append(avg_acc)
        train_log['prec'].append(avg_prec)
        train_log['rec'].append(avg_rec)
        train_log['f1'].append(f1)
        # 验证集
        val_metrics = evaluate_model(model, val_loader, device)
        for k, v in zip(val_log.keys(), val_metrics):
            val_log[k].append(v)
        print(f"Epoch {epoch+1}/{epochs} | train_loss: {avg_loss:.4f} | val_loss: {val_metrics[0]:.4f}")
        # 早停(可选)
        if val_metrics[0] < best_val_loss:
            best_val_loss = val_metrics[0]
            wait = 0
            torch.save(model.state_dict(), 'best_unet_model.pt')
        # else:
        #     wait += 1
        #     if wait >= patience:
        #         print("早停触发，训练停止。")
        #         break
    # 6. 绘制训练曲线
    plot_training_curves(train_log, val_log, ['loss', 'acc', 'prec', 'rec'])
    # 7. 最终评估
    print("\n最终模型评估：")
    best_model = UNet1D().to(device)
    best_model.load_state_dict(torch.load('best_unet_model.pt'))
    evaluate_model(best_model, val_loader, device)

if __name__ == "__main__":
    print("="*50)
    #print(f"PyTorch 版本: {torch.__version__}")
    #print(f"GPU 可用: {torch.cuda.is_available()}")
    # if torch.cuda.is_available():
    #     print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
    # print("="*50)
    train_model()