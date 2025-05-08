def create_empty_tsv(file_path):
    """
    创建一个空的TSV文件，并写入标题行
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # 写入标题行
            f.write("label\ttext_a\n")
        print(f"空的TSV文件已创建: {file_path}")
    except Exception as e:
        print(f"创建文件时出错: {str(e)}")

if __name__ == '__main__':
    new_file_path = "demo21_dataset_fine_tuning/demo21_nolabel_dataset.tsv"
    #new_file_path = "mixed_dataset/valid_sorted.tsv"
   # new_file_path = "mixed_dataset/sorted_cut_valid_dataset.tsv"
    create_empty_tsv(new_file_path)