#!/usr/bin/python3
#-*- coding:utf-8 -*-

import json
import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import numpy as np
import torch
import random
from typing import List, Tuple, Generator, Any
import math

random.seed(40)

# 更新路径配置
dataset_dir = "./ISCXVPN_dataset/"  # ISCXVPN数据集目录
word_dir = "./corpora/"  # 语料库输出目录
word_name = "iscxvpn_corpus.txt"  # 语料库文件名
vocab_dir = "./models/"  
vocab_name = "iscxvpn_vocab.txt"  # 词汇表文件名

# 注释掉所有PCAP相关函数
"""
def pcap_preprocess():
    # ... 原函数内容 ...

def preprocess(pcap_dir):
    # ... 原函数内容 ...

def read_pcap_feature(pcap_file):
    # ... 原函数内容 ...

def read_pcap_flow(pcap_file):
    # ... 原函数内容 ...

def split_cap(pcap_file,pcap_name):
    # ... 原函数内容 ...
"""

def check_data_format(file_path):
    """检查NPY文件的数据格式"""
    data = np.load(file_path)
    print(f"\n检查文件: {file_path}")
    print(f"数据形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    if len(data) > 0:
        print(f"第一个样本类型: {type(data[0])}")
        print(f"第一个样本内容示例: {data[0][:20]}")  # 只显示前20个元素

def process_npy_files():
    """
    处理ISCXVPN数据集中的NPY文件，生成统一的语料库
    """
    corpus = []
    total_samples = 0
    
    # 修正文件名的大小写
    npy_files = {
        'train': 'x_payload_train.npy',
        'test': 'x_payload_test.npy',
        'valid': 'x_payload_valid.npy'
    }
    
    print(f"检查目录 {dataset_dir} 中的文件:")
    existing_files = os.listdir(dataset_dir)
    print("发现以下.npy文件:")
    for file in existing_files:
        if file.endswith('.npy'):
            print(f"- {file}")
    
    print("\n开始处理NPY文件...")
    
    for dataset_type, filename in npy_files.items():
        file_path = os.path.join(dataset_dir, filename)
        if os.path.exists(file_path):
            print(f"正在处理{dataset_type}数据集: {filename}")
            try:
                # 加载NPY文件
                data = np.load(file_path)
                print(f"{dataset_type}数据集形状: {data.shape}")
                
                # 处理每个样本
                for i, sample in enumerate(data):
                    try:
                        # 确保sample是字节类型
                        if isinstance(sample, str):
                            # 如果是字符串，直接使用
                            hex_string = sample
                        else:
                            # 如果是数值数组，转换为十六进制
                            hex_string = ''.join([f'{int(x):02x}' for x in sample if x != 0])
                        
                        # 生成二元组（bigram）文本
                        if hex_string:  # 确保不是空字符串
                            processed_text = bigram_generation(hex_string)
                            if processed_text:  # 确保处理结果不为空
                                corpus.append(processed_text)
                                total_samples += 1
                                
                        if i > 0 and i % 1000 == 0:
                            print(f"已处理 {i} 个样本...")
                            
                    except Exception as e:
                        print(f"处理样本 {i} 时出错: {str(e)}")
                        continue
                
                print(f"完成处理{dataset_type}数据集: 成功处理 {total_samples} 个样本")
                
            except Exception as e:
                print(f"处理{filename}时出错: {str(e)}")
                continue
        else:
            print(f"警告: 文件不存在 - {file_path}")
    
    print(f"\n数据处理完成:")
    print(f"- 总共处理了 {total_samples} 个样本")
    print(f"- 生成了 {len(corpus)} 个语料条目")
    
    # 保存整合后的语料库
    if corpus:
        output_path = os.path.join(word_dir, word_name)
        try:
            with open(output_path, 'w') as f:
                for text in corpus:
                    f.write(text + '\n')
            print(f"语料库已保存到: {output_path}")
        except Exception as e:
            print(f"保存语料库时出错: {str(e)}")
    
    return corpus

def build_BPE():
    # generate source dictionary,0-65535
    num_count = 65536
    not_change_string_count = 5
    i = 0
    source_dictionary = {} 
    tuple_sep = ()
    tuple_cls = ()
    #'PAD':0,'UNK':1,'CLS':2,'SEP':3,'MASK':4
    while i < num_count:
        temp_string = '{:04x}'.format(i) 
        source_dictionary[temp_string] = i
        i += 1
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.WordPiece(vocab=source_dictionary,unk_token="[UNK]",max_input_chars_per_word=4))

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece()
    tokenizer.post_processor = processors.BertProcessing(sep=("[SEP]",1),cls=('[CLS]',2))

    # And then train
    trainer = trainers.WordPieceTrainer(vocab_size=65536, min_frequency=2)
    tokenizer.train([word_dir+word_name, word_dir+word_name], trainer=trainer)

    # And Save it
    tokenizer.save("wordpiece.tokenizer.json", pretty=True)
    return 0

def build_vocab():
    json_file = open("wordpiece.tokenizer.json",'r')
    json_content = json_file.read()
    json_file.close()
    vocab_json = json.loads(json_content)
    vocab_txt = ["[PAD]","[SEP]","[CLS]","[UNK]","[MASK]"]
    for item in vocab_json['model']['vocab']:
        vocab_txt.append(item) # append key of vocab_json
    with open(vocab_dir+vocab_name,'w') as f:
        for word in vocab_txt:
            f.write(word+"\n")
    return 0

def bigram_generation(packet_string,flag=False):
    result = ''
    sentence = cut(packet_string,1)
    token_count = 0
    for sub_string_index in range(len(sentence)):
        if sub_string_index != (len(sentence) - 1):
            token_count += 1
            if token_count > 256: 
                break
            else:
                merge_word_bigram = sentence[sub_string_index] + sentence[sub_string_index + 1]
        else:
            break
        result += merge_word_bigram
        result += ' '
    if flag == True:
        result = result.rstrip()

    return result

def detect_unknown_traffic(text, tokenizer, vocab, threshold=0.8):
    """
    检测未知流量
    Args:
        text: 输入的流量文本
        tokenizer: 分词器
        vocab: 当前词汇表
        threshold: 未知流量判定阈值
    """
    tokens = tokenizer.tokenize(text)
    unknown_count = sum(1 for token in tokens if token not in vocab)
    unknown_ratio = unknown_count / len(tokens)
    return unknown_ratio > threshold

def update_vocab(new_texts, tokenizer, vocab_path):
    """
    更新词汇表
    Args:
        new_texts: 新的未知流量文本列表
        tokenizer: 分词器
        vocab_path: 词汇表路径
    """
    # 读取现有词汇表
    current_vocab = set()
    with open(vocab_path, 'r') as f:
        for line in f:
            current_vocab.add(line.strip())
    
    # 收集新词
    new_tokens = set()
    for text in new_texts:
        tokens = tokenizer.tokenize(text)
        new_tokens.update(tokens)
    
    # 添加新词到词汇表
    new_tokens = new_tokens - current_vocab
    with open(vocab_path, 'a') as f:
        for token in new_tokens:
            f.write(f"{token}\n")

# 添加 cut 函数定义
def cut(obj: str, sec: int) -> List[str]:
    """
    将字符串按指定长度切分
    Args:
        obj: 要切分的字符串
        sec: 切分长度
    Returns:
        切分后的字符串列表
    """
    result = [obj[i:i+sec] for i in range(0,len(obj),sec)]
    remanent_count = len(result[0])%4
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i+sec+remanent_count] for i in range(0,len(obj),sec+remanent_count)]
    return result

# 修改 batch_loader 函数的类型注解
def batch_loader(batch_size: int, *tensors) -> Generator[Tuple[torch.Tensor, ...], None, None]:
    """
    生成批次数据
    Args:
        batch_size: 批次大小
        tensors: 要分批的张量
    Yields:
        批次数据的元组
    """
    length = tensors[0].size(0)
    num_batches = math.ceil(length / batch_size)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, length)
        yield tuple(tensor[start_idx:end_idx] for tensor in tensors)

# 修改 predict 函数，移除 args 依赖
def predict(model, text: str, tokenizer) -> int:
    """
    单个样本预测函数
    """
    # 使用常量替代硬编码的特殊标记
    src = tokenizer.convert_tokens_to_ids([SPECIAL_TOKENS["CLS"]] + tokenizer.tokenize(text))
    seg = [1] * len(src)
    
    # 处理序列长度，使用MODEL_CONFIG中的配置
    if len(src) > MODEL_CONFIG["seq_length"]:
        src = src[: MODEL_CONFIG["seq_length"]]
        seg = seg[: MODEL_CONFIG["seq_length"]]
    while len(src) < MODEL_CONFIG["seq_length"]:
        src.append(0)
        seg.append(0)
    
    # 转换为tensor
    src = torch.LongTensor([src]).to(model.device)
    seg = torch.LongTensor([seg]).to(model.device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        _, logits = model(src, None, seg)
    pred = torch.argmax(logits, dim=1)
    
    return pred.item()

# 修改 retrain_model 函数，移除 args 依赖
def retrain_model(model, new_texts: List[str], tokenizer, 
                 learning_rate: float = 2e-5, 
                 epochs_num: int = 3) -> torch.nn.Module:
    """
    模型重训练函数
    Args:
        model: 现有模型
        new_texts: 新的训练数据
        tokenizer: 分词器
        learning_rate: 学习率
        epochs_num: 训练轮数
    Returns:
        训练后的模型
    """
    # 准备数据
    dataset = []
    for text in new_texts:
        src = tokenizer.convert_tokens_to_ids([SPECIAL_TOKENS["CLS"]] + tokenizer.tokenize(text))
        seg = [1] * len(src)
        if len(src) > MODEL_CONFIG["seq_length"]:
            src = src[: MODEL_CONFIG["seq_length"]]
            seg = seg[: MODEL_CONFIG["seq_length"]]
        while len(src) < MODEL_CONFIG["seq_length"]:
            src.append(0)
            seg.append(0)
        dataset.append((src, seg))
    
    # 转换为tensor
    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])
    
    # 设置训练模式
    model.train()
    
    # 训练循环
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs_num):
        for i, (src_batch, seg_batch) in enumerate(batch_loader(MODEL_CONFIG["batch_size"], src, seg)):
            src_batch = src_batch.to(model.device)
            seg_batch = seg_batch.to(model.device)
            
            # 前向传播和优化
            loss, _ = model(src_batch, None, seg_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    return model

# 修改 incremental_inference 函数，添加必要的参数
def incremental_inference(model, tokenizer, vocab_path, test_path):
    """
    增量学习推理
    """
    unknown_texts = []  # 存储未知流量
    
    # 读取测试数据
    with open(test_path, 'r') as f:
        for line in f:
            text = line.strip()
            
            # 检测是否为未知流量
            if detect_unknown_traffic(text, tokenizer, vocab_path):
                # 标记为未知类别
                unknown_texts.append(text)
                prediction = "UNKNOWN"
            else:
                # 正常预测
                prediction = predict(model, text, tokenizer)
            
            # 输出预测结果
            yield text, prediction
    
    # 如果有未知流量，更新词汇表
    if unknown_texts:
        update_vocab(unknown_texts, tokenizer, vocab_path)
        # 可以选择在这里重新训练模型
        retrain_model(model, unknown_texts, tokenizer)

# 添加特殊标记常量定义
SPECIAL_TOKENS = {
    "PAD": "[PAD]",
    "SEP": "[SEP]",
    "CLS": "[CLS]",
    "UNK": "[UNK]",
    "MASK": "[MASK]"
}

# 模型相关配置
MODEL_CONFIG = {
    "seq_length": 128,
    "batch_size": 32,
    "vocab_size": 65536,
    "min_freq": 2
}

# 修改主函数
if __name__ == '__main__':
    # 确保输出目录存在
    os.makedirs(word_dir, exist_ok=True)
    os.makedirs(vocab_dir, exist_ok=True)
    
    # 检查数据格式
    print("检查数据格式...")
    check_data_format(os.path.join(dataset_dir, 'x_payload_train.npy'))
    
    # 处理NPY文件生成语料库
    corpus = process_npy_files()
    
    if corpus:
        print("\n开始构建词汇表...")
        build_BPE()
        build_vocab()
        print("词汇表构建完成!")
    else:
        print("错误: 语料库为空，无法构建词汇表")
