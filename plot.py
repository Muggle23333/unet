import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.gridspec import GridSpec
from collections import Counter

def plot_dataset_split(X_train, X_val, y_train, y_val):
    """可视化数据集划分情况"""
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 样本长度分布
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist([len(x) for x in X_train], bins=30, alpha=0.7, label='训练集')
    ax1.hist([len(x) for x in X_val], bins=30, alpha=0.7, label='验证集')
    ax1.legend()  # 放在所有hist之后

    ax1.set_title('序列长度分布对比')
    ax1.set_xlabel('序列长度')
    ax1.set_ylabel('频数')
    ax1.legend()
    
    # 标签分布
    ax2 = fig.add_subplot(gs[0, 1:])
    train_counts = Counter(np.concatenate(y_train).flatten())
    val_counts = Counter(np.concatenate(y_val).flatten())
    ax2.bar(train_counts.keys(), train_counts.values(), alpha=0.7, label='训练集')
    ax2.bar(val_counts.keys(), val_counts.values(), alpha=0.7, label='验证集')
    ax2.legend()  # 放在所有hist之后
    ax2.set_title('标签类别分布对比')
    ax2.set_xlabel('class')
    ax2.set_ylabel('count')
    ax2.legend()
    
    # 样本示例可视化
    ax3 = fig.add_subplot(gs[1, :])
    #sample_idx = np.random.randint(0, len(X_train))
    sample_idx = 10
    ax3.plot(X_train[sample_idx], label='地震波形')
    event_points = np.where(y_train[sample_idx] == 1)[0]
    ax3.scatter(event_points, X_train[sample_idx][event_points], 
               color='red', label='地震事件')
    ax3.set_title(f'训练集样本示例 (ID: {sample_idx})')
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('振幅')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('dataset_split.png', dpi=300)
    plt.show()