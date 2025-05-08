# visualize_results.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from datetime import datetime
import torch
import torch.nn as nn
import argparse

class ResultVisualizer:
    def __init__(self, save_dir='visualization_results'):
        """
        初始化可视化器
        """
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
    def load_prediction_data(self, pred_file, test_file):
        """
        加载预测结果和真实标签
        """
        # 添加文件存在检查
        if not os.path.exists(pred_file):
            print(f"错误：预测文件不存在: {pred_file}")
            return None, None, None
        if not os.path.exists(test_file):
            print(f"错误：测试文件不存在: {test_file}")
            return None, None, None
        
        try:
            # 读取预测结果
            predictions = []
            confidences = []
            with open(pred_file, 'r') as f:
                next(f)  # 跳过标题行
                for line in f:
                    parts = line.strip().split('\t')
                    pred = parts[0]
                    conf = float(parts[-1])
                    predictions.append(pred)
                    confidences.append(conf)
            
            # 读取真实标签
            true_labels = []
            with open(test_file, 'r') as f:
                next(f)  # 跳过标题行
                for line in f:
                    label = line.strip().split('\t')[0]
                    true_labels.append(label)
                
            if not predictions or not true_labels:
                print("警告：读取的数据为空")
                return None, None, None
            
            print(f"成功读取数据：{len(predictions)} 个预测结果，{len(true_labels)} 个真实标签")
            return predictions, confidences, true_labels
        
        except Exception as e:
            print(f"读取数据时发生错误: {str(e)}")
            return None, None, None
    
    def plot_confidence_histogram(self, confidences, predictions):
        """
        绘制置信度分布直方图
        """
        plt.figure(figsize=(10, 6))
        known_conf = [conf for pred, conf in zip(predictions, confidences) if pred != "unknown"]
        unknown_conf = [conf for pred, conf in zip(predictions, confidences) if pred == "unknown"]
        
        plt.hist(known_conf, bins=50, alpha=0.5, label='已知流量', density=True)
        plt.hist(unknown_conf, bins=50, alpha=0.5, label='未知流量', density=True)
        
        plt.title('置信度分布直方图')
        plt.xlabel('置信度')
        plt.ylabel('密度')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.save_dir}/confidence_histogram.png')
        plt.close()
    
    def plot_confidence_boxplot(self, confidences, predictions, true_labels):
        """
        绘制置信度箱型图
        """
        plt.figure(figsize=(12, 6))
        data = []
        labels = []
        
        # 已知vs未知的箱型图
        known_conf = [conf for pred, conf in zip(predictions, confidences) if pred != "unknown"]
        unknown_conf = [conf for pred, conf in zip(predictions, confidences) if pred == "unknown"]
        
        plt.subplot(1, 2, 1)
        plt.boxplot([known_conf, unknown_conf], labels=['已知流量', '未知流量'])
        plt.title('已知/未知流量置信度分布')
        plt.ylabel('置信度')
        
        # 各类别的箱型图
        plt.subplot(1, 2, 2)
        unique_labels = sorted(set(true_labels))
        class_conf = []
        for label in unique_labels:
            label_conf = [conf for true, conf in zip(true_labels, confidences) if true == label]
            class_conf.append(label_conf)
        
        plt.boxplot(class_conf, labels=unique_labels)
        plt.title('各类别置信度分布')
        plt.xlabel('类别')
        plt.ylabel('置信度')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/confidence_boxplots.png')
        plt.close()
    
    def plot_confusion_matrix(self, true_labels, predictions):
        """
        绘制混淆矩阵热力图
        """
        plt.figure(figsize=(10, 8))
        
        # 创建混淆矩阵
        unique_true = sorted(set(true_labels))
        unique_pred = sorted(set(predictions))
        cm = np.zeros((len(unique_true), len(unique_pred)))
        
        label_to_idx_true = {label: i for i, label in enumerate(unique_true)}
        label_to_idx_pred = {label: i for i, label in enumerate(unique_pred)}
        
        for t, p in zip(true_labels, predictions):
            cm[label_to_idx_true[t]][label_to_idx_pred[p]] += 1
        
        # 绘制热力图
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='YlOrRd',
                    xticklabels=unique_pred, yticklabels=unique_true)
        plt.title('预测结果混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.savefig(f'{self.save_dir}/confusion_matrix.png')
        plt.close()
    
    def plot_threshold_impact(self, confidences):
        """
        绘制置信度阈值影响曲线
        """
        plt.figure(figsize=(10, 6))
        thresholds = np.linspace(0, 1, 100)
        unknown_ratios = []
        for threshold in thresholds:
            unknown_count = sum(1 for conf in confidences if conf < threshold)
            unknown_ratios.append(unknown_count / len(confidences))
        
        plt.plot(thresholds, unknown_ratios)
        plt.title('置信度阈值与未知流量检测率关系')
        plt.xlabel('置信度阈值')
        plt.ylabel('未知流量占比')
        plt.grid(True)
        plt.savefig(f'{self.save_dir}/threshold_impact.png')
        plt.close()
    
    def calculate_all_confidences(self, logits):
        """
        同时计算三种置信度方法的结果
        """
        prob = nn.Softmax(dim=1)(logits)
        results = {}
        
        # 方法1：最大概率
        max_conf, pred = torch.max(prob, dim=1)
        results['max_prob'] = {'confidence': max_conf, 'pred': pred}
        
        # 方法2：边际差值
        sorted_prob, _ = torch.sort(prob, dim=1, descending=True)
        margin_conf = sorted_prob[:, 0] - sorted_prob[:, 1]
        results['margin'] = {'confidence': margin_conf, 'pred': pred}
        
        # 方法3：熵
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
        entropy_conf = 1 - entropy / np.log(prob.size(1))
        results['entropy'] = {'confidence': entropy_conf, 'pred': pred}
        
        return results
    
    def normalize_confidence(self, confidences):
        """
        将置信度值归一化到[0,1]区间
        """
        min_conf = np.min(confidences)
        max_conf = np.max(confidences)
        if max_conf == min_conf:
            return np.ones_like(confidences)
        return (confidences - min_conf) / (max_conf - min_conf)
    
    def plot_confidence_comparison(self, all_confidences, true_labels, predictions, unknown_label="8"):
        """
        比较三种置信度方法的效果
        """
        plt.figure(figsize=(15, 10))
        
        # 1. 对每种方法的置信度进行归一化
        normalized_confidences = {}
        for method in ['max_prob', 'margin', 'entropy']:
            normalized_confidences[method] = self.normalize_confidence(all_confidences[method]['confidence'])
        
        # 2. 绘制已知类别和未知类别的置信度分布
        plt.subplot(2, 1, 1)
        for method in ['max_prob', 'margin', 'entropy']:
            # 获取未知类别的置信度
            unknown_conf = [conf for conf in normalized_confidences[method] if conf == 0]
            # 获取已知类别的置信度
            known_conf = [conf for conf in normalized_confidences[method] if conf == 1]
            
            # 绘制密度图
            sns.kdeplot(unknown_conf, label=f'{method}-未知类别', linestyle='--')
            sns.kdeplot(known_conf, label=f'{method}-已知类别')
        
        plt.title('三种方法的置信度分布比较（归一化后）')
        plt.xlabel('归一化置信度')
        plt.ylabel('密度')
        plt.legend()
        
        # 3. 绘制ROC曲线
        plt.subplot(2, 1, 2)
        for method in ['max_prob', 'margin', 'entropy']:
            confidences = normalized_confidences[method]
            # 计算真实的已知/未知标签
            true_unknown = np.array([1 if label == unknown_label else 0 for label in true_labels])
            
            # 计算不同阈值下的TPR和FPR
            thresholds = np.linspace(0, 1, 100)
            tpr = []
            fpr = []
            for threshold in thresholds:
                pred_unknown = (confidences < threshold).astype(int)
                tp = np.sum((pred_unknown == 1) & (true_unknown == 1))
                fp = np.sum((pred_unknown == 1) & (true_unknown == 0))
                tn = np.sum((pred_unknown == 0) & (true_unknown == 0))
                fn = np.sum((pred_unknown == 0) & (true_unknown == 1))
                
                tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
            
            # 计算AUC
            auc = np.trapz(tpr, fpr)
            plt.plot(fpr, tpr, label=f'{method} (AUC={auc:.3f})')
        
        plt.title('未知类别检测的ROC曲线比较')
        plt.xlabel('假阳性率 (FPR)')
        plt.ylabel('真阳性率 (TPR)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/confidence_methods_comparison.png')
        plt.close()
    
    def visualize_all(self, pred_files, test_file):
        """
        生成所有可视化图表，包括三种方法的对比
        """
        print(f"开始处理数据...")
        print(f"预测文件: {pred_files}")
        print(f"测试文件: {test_file}")
        
        # 存储三种方法的数据
        method_data = {}
        method_names = ['max_prob', 'margin', 'entropy']
        
        # 读取所有预测文件的数据
        for method, pred_file in zip(method_names, pred_files):
            predictions, confidences, true_labels = self.load_prediction_data(pred_file, test_file)
            if predictions is None or confidences is None or true_labels is None:
                print(f"无法加载{method}方法的数据")
                return
            method_data[method] = {
                'predictions': predictions,
                'confidences': confidences,
                'true_labels': true_labels
            }
        
        print(f"开始生成可视化图表...")
        try:
            # 生成单个方法的基础图表（使用max_prob方法的结果）
            base_method = 'max_prob'
            self.plot_confidence_histogram(
                method_data[base_method]['confidences'], 
                method_data[base_method]['predictions']
            )
            print("- 已生成置信度分布直方图")
            
            self.plot_confidence_boxplot(
                method_data[base_method]['confidences'],
                method_data[base_method]['predictions'],
                method_data[base_method]['true_labels']
            )
            print("- 已生成置信度箱型图")
            
            self.plot_confusion_matrix(
                method_data[base_method]['true_labels'],
                method_data[base_method]['predictions']
            )
            print("- 已生成混淆矩阵热力图")
            
            self.plot_threshold_impact(method_data[base_method]['confidences'])
            print("- 已生成阈值影响曲线")
            
            # 生成三种方法的对比图
            self.plot_methods_comparison(method_data)
            print("- 已生成三种方法对比图")
            
            print(f"\n所有可视化结果已保存至目录: {self.save_dir}")
            
        except Exception as e:
            print(f"生成可视化图表时发生错误: {str(e)}")

    def plot_methods_comparison(self, method_data):
        """
        绘制三种方法的对比图
        """
        plt.figure(figsize=(15, 12))
        
        # 1. 置信度分布对比
        plt.subplot(2, 1, 1)
        for method in method_data.keys():
            confidences = method_data[method]['confidences']
            predictions = method_data[method]['predictions']
            
            known_conf = [conf for pred, conf in zip(predictions, confidences) if pred != "unknown"]
            unknown_conf = [conf for pred, conf in zip(predictions, confidences) if pred == "unknown"]
            
            sns.kdeplot(known_conf, label=f'{method}-已知类别')
            sns.kdeplot(unknown_conf, label=f'{method}-未知类别', linestyle='--')
        
        plt.title('三种方法的置信度分布对比')
        plt.xlabel('置信度')
        plt.ylabel('密度')
        plt.legend()
        plt.grid(True)
        
        # 2. ROC曲线对比
        plt.subplot(2, 1, 2)
        for method in method_data.keys():
            confidences = method_data[method]['confidences']
            predictions = method_data[method]['predictions']
            true_labels = method_data[method]['true_labels']
            
            # 计算真实的已知/未知标签
            true_unknown = np.array([1 if label == "unknown" else 0 for label in true_labels])
            
            # 计算不同阈值下的TPR和FPR
            thresholds = np.linspace(0, 1, 100)
            tpr = []
            fpr = []
            for threshold in thresholds:
                pred_unknown = (np.array(confidences) < threshold).astype(int)
                tp = np.sum((pred_unknown == 1) & (true_unknown == 1))
                fp = np.sum((pred_unknown == 1) & (true_unknown == 0))
                tn = np.sum((pred_unknown == 0) & (true_unknown == 0))
                fn = np.sum((pred_unknown == 0) & (true_unknown == 1))
                
                tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
            
            # 计算AUC
            auc = np.trapz(tpr, fpr)
            plt.plot(fpr, tpr, label=f'{method} (AUC={auc:.3f})')
        
        plt.title('三种方法的ROC曲线对比')
        plt.xlabel('假阳性率 (FPR)')
        plt.ylabel('真阳性率 (TPR)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/methods_comparison.png')
        plt.close()

def main():
    """
    独立运行时的入口函数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_files', nargs='+', required=True, 
                       help='不同方法的预测结果文件路径列表')
    parser.add_argument('--test_file', type=str, required=True,
                       help='测试集文件路径')
    parser.add_argument('--save_dir', type=str, default='visualization_results',
                       help='可视化结果保存目录')
    args = parser.parse_args()
    
    visualizer = ResultVisualizer(save_dir=args.save_dir)
    visualizer.visualize_all(args.pred_files, args.test_file)

if __name__ == '__main__':
    main()