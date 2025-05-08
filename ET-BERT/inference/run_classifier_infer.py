"""
  This script provides an exmaple to wrap UER-py for classification inference.
"""
import sys
import os
import torch
import argparse
import collections
import torch.nn as nn
import numpy as np

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

# 确保fine-tuning目录在Python路径中
fine_tuning_dir = os.path.join(uer_dir, "fine-tuning")
if fine_tuning_dir not in sys.path:
    sys.path.append(fine_tuning_dir)

from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_loader import load_model
from uer.opts import infer_opts

from run_classifier import Classifier


def batch_loader(batch_size, src, seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        yield src_batch, seg_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        yield src_batch, seg_batch


def read_dataset(args, path):
    """
    推理阶段的数据读取函数
    直接读取文本内容进行预测，不需要标签信息
    每行一个文本样本
    """
    dataset = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:  # 跳过空行
                continue
            
            # 文本预处理：添加[CLS]标记并转换为id
            src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text))
            seg = [1] * len(src)
            
            # 处理序列长度
            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)  # padding
                seg.append(0)
                
            dataset.append((src, seg))
            
    print(f"读取了 {len(dataset)} 个待预测样本")
    return dataset


def calculate_confidence(logits, method='max_prob'):
    """
    计算置信度的多种方法
    """
    prob = nn.Softmax(dim=1)(logits)
    
    if method == 'max_prob':
        # 方法1：最大概率值（当前使用的方法）
        confidence, pred = torch.max(prob, dim=1)
    elif method == 'margin':
        # 方法2：最大概率值与第二大概率值的差
        sorted_prob, _ = torch.sort(prob, dim=1, descending=True)
        confidence = sorted_prob[:, 0] - sorted_prob[:, 1]
        pred = torch.argmax(prob, dim=1)
    elif method == 'entropy':
        # 方法3：预测熵的负值（越大表示越确定）
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
        confidence = 1 - entropy / np.log(prob.size(1))
        pred = torch.argmax(prob, dim=1)
    
    return confidence, pred


def evaluate_predictions(pred_file, test_file):
    """
    评估预测结果的准确率和F1分数
    """
    try:
        # 读取预测结果和真实标签
        predictions = []
        confidences = []  # 添加置信度列表
        with open(pred_file, 'r') as f:
            next(f)  # 跳过标题行
            for line in f:
                parts = line.strip().split('\t')
                pred = parts[0]  # 预测标签
                conf = float(parts[-1])  # 置信度在最后一列
                predictions.append(pred)
                confidences.append(conf)
                
        true_labels = []
        with open(test_file, 'r') as f:
            next(f)  # 跳过标题行
            for line in f:
                label = line.strip().split('\t')[0]
                true_labels.append(label)
        
        if len(predictions) != len(true_labels):
            print(f"警告：预测数量({len(predictions)})与实际标签数量({len(true_labels)})不匹配")
            return None
        
        # 首先打印未知流量的统计信息
        unknown_count = sum(1 for p in predictions if p == "unknown")
        known_count = len(predictions) - unknown_count
        print("\n" + "="*20 + " 未知流量统计 " + "="*20)
        print(f"总样本数: {len(predictions)}")
        print(f"未知流量数量: {unknown_count}")
        print(f"未知流量占比: {unknown_count/len(predictions):.2%}")
        
        # 计算已知流量的准确率统计
        print("\n" + "="*20 + " 已知流量评估 " + "="*20)
        print(f"已知流量样本数: {known_count}")
        
        # 计算整体指标
        y_true = np.array(true_labels)
        y_pred = np.array(predictions)
        
        # 计算已知样本的指标
        known_mask = y_pred != "unknown"
        known_true = y_true[known_mask]
        known_pred = y_pred[known_mask]
        
        # 计算每个类别的指标
        labels = sorted(set(true_labels))
        print("\n各类别评估指标:")
        print("-" * 70)
        print(f"{'类别':^6} | {'准确率':^8} | {'召回率':^8} | {'F1分数':^8} | {'样本数':^8} | {'未知数':^8}")
        print("-" * 70)
        
        overall_metrics = {'precision': [], 'recall': [], 'f1': []}
        
        for label in labels:
            # 获取该类别的所有样本索引
            label_mask = y_true == label
            label_total = np.sum(label_mask)
            
            # 计算未知样本数
            unknown_count = np.sum((y_pred == "unknown") & label_mask)
            
            # 计算已知样本的指标
            known_label_mask = label_mask & known_mask
            true_positive = np.sum((y_pred == label) & label_mask)
            false_positive = np.sum((y_pred == label) & ~label_mask)
            false_negative = np.sum((y_pred != label) & (y_pred != "unknown") & label_mask)
            
            # 计算精确率、召回率和F1分数
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            overall_metrics['precision'].append(precision)
            overall_metrics['recall'].append(recall)
            overall_metrics['f1'].append(f1)
            
            print(f"{label:^6} | {precision:^8.4f} | {recall:^8.4f} | {f1:^8.4f} | {label_total:^8d} | {unknown_count:^8d}")
        
        print("-" * 70)
        
        # 计算宏平均指标
        macro_precision = np.mean(overall_metrics['precision'])
        macro_recall = np.mean(overall_metrics['recall'])
        macro_f1 = np.mean(overall_metrics['f1'])
        
        print("\n宏平均指标:")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        
        return macro_f1
        
    except Exception as e:
        print(f"评估过程发生错误: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--labels_num", type=int, required=True,
                        help="Number of prediction labels.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    parser.add_argument("--output_logits", action="store_true", help="Write logits to output file.")
    parser.add_argument("--output_prob", action="store_true", help="Write probabilities to output file.")
    
    # 添加评估参数
    parser.add_argument("--eval", action="store_true", help="是否评估预测结果")
    parser.add_argument("--test_label_path", type=str, help="带标签的测试集文件路径，用于评估")
    
    # 添加置信度阈值参数
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="置信度阈值，低于此值的预测将被标记为未知流量")
    parser.add_argument("--unknown_label", type=str, default="unknown",
                        help="未知流量的标签")
    
    # 在main函数中添加置信度计算方法参数
    parser.add_argument("--confidence_method", choices=['max_prob', 'margin', 'entropy'], 
                        default='max_prob', help="置信度计算方法")
    
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model and load parameters.
    args.soft_targets, args.soft_alpha = False, False
    model = Classifier(args)
    model = load_model(model, args.load_model_path)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    dataset = read_dataset(args, args.test_path)

    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])

    batch_size = args.batch_size
    instances_num = src.size()[0]

    print("The number of prediction instances: ", instances_num)

    # 初始化计数器
    known_count = 0
    unknown_count = 0

    model.eval()

    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        f.write("label")
        if args.output_logits:
            f.write("\t" + "logits")
        if args.output_prob:
            f.write("\t" + "prob")
        f.write("\t" + "confidence")  # 添加置信度列
        f.write("\n")
        
        for i, (src_batch, seg_batch) in enumerate(batch_loader(batch_size, src, seg)):
            src_batch = src_batch.to(device)
            seg_batch = seg_batch.to(device)
            with torch.no_grad():
                _, logits = model(src_batch, None, seg_batch)
            
            # 在预测部分使用新的置信度计算函数
            confidence_scores, pred = calculate_confidence(logits, method=args.confidence_method)
            
            # 转换为numpy数组
            pred = pred.cpu().numpy()
            confidence_scores = confidence_scores.cpu().numpy()
            
            # 保持logits为tensor以便计算概率
            prob = nn.Softmax(dim=1)(logits)
            
            # 现在再转换为numpy和list
            logits_np = logits.cpu().numpy()
            prob_np = prob.cpu().numpy()
            
            for j in range(len(pred)):
                # 根据置信度判断是否为未知流量
                if confidence_scores[j] < args.confidence_threshold:
                    label = args.unknown_label
                    unknown_count += 1
                else:
                    label = str(pred[j])
                    known_count += 1
                
                f.write(label)
                if args.output_logits:
                    f.write("\t" + " ".join([str(v) for v in logits_np[j]]))
                if args.output_prob:
                    f.write("\t" + " ".join([str(v) for v in prob_np[j]]))
                f.write("\t" + str(confidence_scores[j]))  # 写入置信度
                f.write("\n")

    # 打印统计信息
    print("\n" + "="*20 + " 预测统计 " + "="*20)
    print(f"总样本数: {instances_num}")
    print(f"已知流量数量: {known_count}")
    print(f"未知流量数量: {unknown_count}")
    print(f"未知流量占比: {unknown_count/instances_num:.2%}")

    # 如果指定了评估，计算准确率
    if args.eval and args.test_label_path:
        evaluate_predictions(args.prediction_path, args.test_label_path)


if __name__ == "__main__":
    main()
