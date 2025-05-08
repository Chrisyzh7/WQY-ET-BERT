import os
import random

def shuffle_and_append(tsv_file_path, output_file_path):
    """
    读取TSV文件，打乱其内容，并追加到另一个文件中
    """
    print(f"开始处理文件: {tsv_file_path}")
    
    if not os.path.exists(tsv_file_path):
        print(f"错误: 找不到文件 {tsv_file_path}")
        return
    
    # 读取TSV文件并跳过标题行
    data = []
    try:
        with open(tsv_file_path, 'r', encoding='utf-8') as f:
            next(f)  # 跳过标题行
            for line in f:
                data.append(line.strip())
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return
    
    # 打乱数据
    random.shuffle(data)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 写入打乱后的数据到一个文件
    try:
        # 检查输出文件是否存在
        file_exists = os.path.exists(output_file_path)
        
        with open(output_file_path, 'a', encoding='utf-8') as f:
            if not file_exists:
                # 如果文件不存在，写入标题行
                f.write("label\ttext_a\n")
            for line in data:
                f.write(f"{line}\n")
        print(f"打乱后的数据已追加到: {output_file_path}")
    except Exception as e:
        print(f"写入文件时出错: {str(e)}")

if __name__ == '__main__':
    input_file = "mixed_dataset/sorted_cut_train_dataset.tsv"
    output_file = "demo21_dataset_fine_tuning/demo21_train_dataset.tsv"
    shuffle_and_append(input_file, output_file)