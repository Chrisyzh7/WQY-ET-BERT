import os

def sort_by_label(tsv_file_path, output_file_path):
    """
    将TSV文件中的数据按标签排序，并保存到新的文件中
    """
    print(f"开始处理文件: {tsv_file_path}")
    
    if not os.path.exists(tsv_file_path):
        print(f"错误: 找不到文件 {tsv_file_path}")
        return
    
    # 读取TSV文件
    data = []
    try:
        with open(tsv_file_path, 'r', encoding='utf-8') as f:
            next(f)  # 跳过标题行
            for line in f:
                parts = line.strip().split('\t')  # 使用制表符分隔
                if len(parts) != 2:
                    print(f"警告: 行格式不正确: {line}")
                    continue
                label, text = parts
                try:
                    data.append((int(label), text))  # 将标签转换为整数
                except ValueError:
                    print(f"警告: 标签不是有效的整数: {label}")
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return
    
    # 按标签排序
    data.sort(key=lambda x: x[0])
    
    # 写入排序后的数据到一个文件
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for label, text in data:
                f.write(f"{label}\t{text}\n")
        print(f"排序后的数据已保存到: {output_file_path}")
    except Exception as e:
        print(f"写入文件时出错: {str(e)}")

if __name__ == '__main__':
    input_file = "dataset_fine_tuning/valid_dataset.tsv"
    output_file = "mixed_dataset/valid_sorted.tsv"
    sort_by_label(input_file, output_file)