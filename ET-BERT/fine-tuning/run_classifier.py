"""
This script provides an exmaple to wrap UER-py for classification.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
import tqdm
import numpy as np

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        # 添加词汇表大小匹配验证
        print("=== Vocabulary Size Verification ===")
        print("Tokenizer vocab size:", len(args.tokenizer.vocab))
        print("Embedding weight size:", self.embedding.word_embedding.weight.size(0))
        print("================================")
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # 只在第一个批次输出信息
        if not hasattr(self, 'printed_debug_info'):
            print("=== Debug Information ===")
            print("Input shape:", src.shape)
            print("Max position index:", src.shape[1])
            print("Max value in src:", src.max().item())
            print("Word embedding size:", self.embedding.word_embedding.weight.size(0))
            print("========================")
            self.printed_debug_info = True
        
        # Embedding.
        emb = self.embedding(src, seg)
        
        # Encoder.
        output = self.encoder(emb, seg)
        temp_output = output
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits
            #return temp_output, logits

        # 添加范围检查
        if not hasattr(self, 'checked_ranges'):
            print("=== Embedding Range Check ===")
            print("Word embedding range:", self.embedding.word_embedding.weight.size(0))
            print("Position embedding range:", self.embedding.position_embedding.weight.size(0))
            print("Segment embedding range:", self.embedding.segment_embedding.weight.size(0))
            print("Input max token id:", src.max().item())
            print("Input max position:", src.size(1))
            print("Input max segment:", seg.max().item())
            print("===========================")
            self.checked_ranges = True


def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location='cpu'), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def read_dataset(args, path, max_samples=None):
    dataset, columns = [], {}
    sample_count = 0
    
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
                
            # 检查是否已达到最大样本数
            if max_samples is not None and sample_count >= max_samples:
                break
                
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            if "text_b" not in columns:  # Sentence classification.
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
                seg = [1] * len(src)
            else:  # Sentence-pair classification.
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)  # 填充索引
                seg.append(0)

            # 只在异常情况下输出警告
            if max(src) >= len(args.tokenizer.vocab):
                print(f"Warning: Token index out of vocabulary range in line {line_id}")
                print(f"Max index: {max(src)}, Vocab size: {len(args.tokenizer.vocab)}")
            
            # 检查序列长度是否异常
            if len(src) > args.seq_length * 1.5:  # 如果序列长度显著超过限制
                print(f"Warning: Extremely long sequence in line {line_id}")
                print(f"Sequence length: {len(src)}, Max length: {args.seq_length}")
            
            # 检查特殊标记
            if src[0] != args.tokenizer.convert_tokens_to_ids([CLS_TOKEN])[0]:
                print(f"Warning: Missing CLS token in line {line_id}")

            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, tgt, seg, soft_tgt))
            else:
                dataset.append((src, tgt, seg))
                
            sample_count += 1
            
    print(f"Read {sample_count} samples from {path}")
    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    
    loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def create_dataloader(args, dataset, batch_size, shuffle=True):
    """
    创建优化的数据加载器，根据设备类型选择合适的配置
    """
    # 提取数据
    if len(dataset[0]) == 4:  # 包含soft_tgt
        src = torch.LongTensor([sample[0] for sample in dataset])
        tgt = torch.LongTensor([sample[1] for sample in dataset])
        seg = torch.LongTensor([sample[2] for sample in dataset])
        soft_tgt = torch.FloatTensor([sample[3] for sample in dataset])
        tensor_dataset = TensorDataset(src, tgt, seg, soft_tgt)
    else:  # 不包含soft_tgt
        src = torch.LongTensor([sample[0] for sample in dataset])
        tgt = torch.LongTensor([sample[1] for sample in dataset])
        seg = torch.LongTensor([sample[2] for sample in dataset])
        tensor_dataset = TensorDataset(src, tgt, seg)
    
    # 根据设备类型选择合适的DataLoader配置
    if args.device.type == 'cuda':
        # GPU配置
        return DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,  # GPU训练使用多个工作进程
            pin_memory=True  # 将数据固定在内存中，加速GPU访问
        )
    else:
        # CPU配置
        return DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,  # CPU训练使用较少的工作进程
            pin_memory=False
        )

def evaluate_with_dataloader(args, dataloader):
    """
    使用DataLoader进行评估
    """
    correct = 0
    total = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()

    for batch in dataloader:
        if len(batch) == 4:  # 包含soft_tgt
            src_batch, tgt_batch, seg_batch, _ = batch
        else:  # 不包含soft_tgt
            src_batch, tgt_batch, seg_batch = batch
            
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        
        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch)
        
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()
        total += pred.size()[0]

    # 打印评估结果
    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / total, correct, total))
    return correct / total, confusion

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")
                        
    # 添加数据集大小限制参数
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Maximum number of training samples to use (for testing/debugging).")
    parser.add_argument("--max_dev_samples", type=int, default=None,
                        help="Maximum number of dev samples to use (for testing/debugging).")
    parser.add_argument("--max_test_samples", type=int, default=None,
                        help="Maximum number of test samples to use (for testing/debugging).")
    
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # 添加设备检测和设置
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU for training")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU for training")
    args.device = device

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # 在训练开始前一次性读取所有数据集
    print("Loading datasets...")
    trainset = read_dataset(args, args.train_path, args.max_train_samples)
    devset = read_dataset(args, args.dev_path, args.max_dev_samples)
    if args.test_path is not None:
        testset = read_dataset(args, args.test_path, args.max_test_samples)
    else:
        testset = None
    
    # 创建数据加载器
    print("Creating data loaders...")
    batch_size = args.batch_size
    train_loader = create_dataloader(args, trainset, batch_size, shuffle=True)
    dev_loader = create_dataloader(args, devset, batch_size, shuffle=False)
    if testset is not None:
        test_loader = create_dataloader(args, testset, batch_size, shuffle=False)
    
    # 计算训练步数
    args.train_steps = len(train_loader) * args.epochs_num

    print("Batch size: ", batch_size)
    print("The number of training instances:", len(trainset))
    print("The number of training steps per epoch:", len(train_loader))
    print("Total training steps:", args.train_steps)

    # Build classification model.
    model = Classifier(args)

    # Load parameters
    if args.pretrained_model_path is not None:
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location=device), strict=False)
    
    model = model.to(device)
    args.model = model

    optimizer, scheduler = build_optimizer(args, model)

    total_loss, best_result = 0.0, 0.0

    print("Start training.")

    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        model.train()
        for i, batch in enumerate(train_loader):
            if len(batch) == 4:  # 包含soft_tgt
                src_batch, tgt_batch, seg_batch, soft_tgt_batch = batch
            else:  # 不包含soft_tgt
                src_batch, tgt_batch, seg_batch = batch
                soft_tgt_batch = None
                
            src_batch = src_batch.to(args.device)
            tgt_batch = tgt_batch.to(args.device)
            seg_batch = seg_batch.to(args.device)
            if soft_tgt_batch is not None:
                soft_tgt_batch = soft_tgt_batch.to(args.device)
            
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            total_loss += loss.item()
            
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        # 使用验证集评估模型
        print(f"Evaluating on dev set after epoch {epoch}...")
        result, _ = evaluate_with_dataloader(args, dev_loader)
        
        if result > best_result:
            best_result = result
            save_model(model, args.output_model_path)
            print(f"New best result: {best_result:.4f}, model saved to {args.output_model_path}")

    # Evaluation phase.
    if testset is not None:
        print("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path))
        evaluate_with_dataloader(args, test_loader)

    # 确保 max_seq_length 至少等于 seq_length
    if not hasattr(args, 'max_seq_length'):
        args.max_seq_length = args.seq_length
    elif args.max_seq_length < args.seq_length:
        args.max_seq_length = args.seq_length

    # 设置默认的 hidden_size
    if not hasattr(args, 'hidden_size'):
        args.hidden_size = 768  # BERT 的默认隐藏层大小


if __name__ == "__main__":
    main()
