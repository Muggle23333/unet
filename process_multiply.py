import os
import pandas as pd

# 定义输入和输出文件夹
input_folder = "test_data_new"
output_folder = "test_data_new_processed"

os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹下所有csv文件
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # 读取csv文件
        df = pd.read_csv(input_path)
        
        # 检查是否有至少两列
        if df.shape[1] < 2:
            print(f"文件 {filename} 列数不足两列，跳过处理。")
            continue
        
        # 计算前两列相乘，并加入到第三列
        df[df.columns[2] if df.shape[1] > 2 else '乘积'] = df.iloc[:, 0] * df.iloc[:, 1]
        
        # 保存结果到新文件夹
        df.to_csv(output_path, index=False)
        print(f"处理完成: {filename} -> {output_path}")