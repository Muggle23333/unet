import os
import pandas as pd

# 输入和输出文件夹
input_folder = 'test_data2'
output_folder = 'test_data_new'

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

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
    arr[first:last+1] = 1
    return pd.Series(arr, index=series.index)

for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path, header=None)
        # 只对第二列处理
        df[1] = process_second_col(df[1])
        # 保存
        output_path = os.path.join(output_folder, filename)
        df.to_csv(output_path, header=False, index=False)