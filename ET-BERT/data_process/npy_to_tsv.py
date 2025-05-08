#!/usr/bin/python3
#-*- coding:utf-8 -*-

import numpy as np
import os
import csv
import sys

# 设置路径
# 输入NPY文件路径
npy_dir = "D:\\wang_BERT\\BERT-Forest\\ET-BERT\\ISCXVPN_dataset_app\\"
# 输出TSV文件路径
tsv_dir = "D:\\wang_BERT\\BERT-Forest\\ET-BERT\\ISCXVPN_dataset_app\\"

def write_dataset_tsv(data, label, file_dir, type):
    """
    将数据和标签写入TSV文件
    
    参数:
    data -- 数据数组
    label -- 标签数组
    file_dir -- 输出目录
    type -- 数据集类型 (train/valid/test)
    """
    dataset_file = [["label", "text_a"]]
    
    # 遍历标签列表，将每个标签和对应的文本数据添加到dataset_file列表中
    for index in range(len(label)):
        dataset_file.append([label[index], data[index]])
    
    # 确保输出目录存在
    os.makedirs(file_dir, exist_ok=True)
    
    # 将dataset_file列表写入到文件中
    output_file = os.path.join(file_dir, f"{type}_dataset.tsv")
    with open(output_file, 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerows(dataset_file)
    
    print(f"已生成 {output_file}")
    return output_file

def unlabel_data(label_data_path):
    """
    从带标签的数据集中提取无标签的文本数据，并保存为新的文件
    
    参数:
    label_data_path -- 带标签的TSV文件路径
    """
    nolabel_data = ""
    with open(label_data_path, newline='') as f:
        data = csv.reader(f, delimiter='\t')
        next(data)  # 跳过表头
        for row in data:
            if len(row) > 1:
                nolabel_data += row[1] + '\n'
    
    nolabel_file = label_data_path.replace("test_dataset", "nolabel_test_dataset")
    with open(nolabel_file, 'w', newline='') as f:
        f.write(nolabel_data)
    
    print(f"已生成无标签文件 {nolabel_file}")
    return nolabel_file

def convert_npy_to_tsv(npy_dir, tsv_dir):
    """
    将NPY文件转换为TSV文件
    
    参数:
    npy_dir -- NPY文件目录
    tsv_dir -- TSV文件输出目录
    """
    try:
        # 加载NPY文件
        print(f"正在从 {npy_dir} 读取NPY文件...")
        
        # 加载特征数据
        x_train_path = os.path.join(npy_dir, 'x_payload_train.npy')
        x_test_path = os.path.join(npy_dir, 'x_payload_test.npy')
        x_valid_path = os.path.join(npy_dir, 'x_payload_valid.npy')
        
        # 加载标签数据
        y_train_path = os.path.join(npy_dir, 'y_train.npy')
        y_test_path = os.path.join(npy_dir, 'y_test.npy')
        y_valid_path = os.path.join(npy_dir, 'y_valid.npy')
        
        # 检查文件是否存在
        for path in [x_train_path, x_test_path, x_valid_path, y_train_path, y_test_path, y_valid_path]:
            if not os.path.exists(path):
                print(f"错误: 文件 {path} 不存在!")
                return False
        
        # 加载数据
        x_train = np.load(x_train_path, allow_pickle=True)
        x_test = np.load(x_test_path, allow_pickle=True)
        x_valid = np.load(x_valid_path, allow_pickle=True)
        y_train = np.load(y_train_path, allow_pickle=True)
        y_test = np.load(y_test_path, allow_pickle=True)
        y_valid = np.load(y_valid_path, allow_pickle=True)
        
        print("数据加载完成，开始生成TSV文件...")
        
        # 生成TSV文件
        train_file = write_dataset_tsv(x_train, y_train, tsv_dir, "train")
        test_file = write_dataset_tsv(x_test, y_test, tsv_dir, "test")
        valid_file = write_dataset_tsv(x_valid, y_valid, tsv_dir, "valid")
        
        # 生成无标签测试文件
        unlabel_file = unlabel_data(test_file)
        
        print("所有文件生成完成!")
        print(f"训练集: {train_file}")
        print(f"测试集: {test_file}")
        print(f"验证集: {valid_file}")
        print(f"无标签测试集: {unlabel_file}")
        
        return True
        
    except Exception as e:
        print(f"转换过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 如果命令行提供了参数，使用命令行参数
    if len(sys.argv) > 2:
        npy_dir = sys.argv[1]
        tsv_dir = sys.argv[2]
    
    print(f"NPY文件目录: {npy_dir}")
    print(f"TSV文件输出目录: {tsv_dir}")
    
    # 执行转换
    convert_npy_to_tsv(npy_dir, tsv_dir) 