"""
数据集模块
实现PyTorch Dataset用于训练和评估
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import sentencepiece as spm


class TranslationDataset(Dataset):
    """
    翻译任务数据集
    
    参数:
        src_file: 源语言文件路径
        tgt_file: 目标语言文件路径
        src_tokenizer: 源语言分词器 (SentencePiece模型)
        tgt_tokenizer: 目标语言分词器
        max_length: 最大序列长度 (默认512)
    """
    
    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        src_tokenizer: spm.SentencePieceProcessor,
        tgt_tokenizer: spm.SentencePieceProcessor,
        max_length: int = 512
    ):
        self.max_length = max_length
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        
        # 读取数据
        self.src_sentences = self._load_file(src_file)
        self.tgt_sentences = self._load_file(tgt_file)
        
        assert len(self.src_sentences) == len(self.tgt_sentences), \
            "源语言和目标语言句子数量不匹配"
        
        # 特殊token ID
        self.pad_id = src_tokenizer.pad_id()
        self.bos_id = src_tokenizer.bos_id()
        self.eos_id = src_tokenizer.eos_id()
    
    def _load_file(self, file_path: str) -> List[str]:
        """
        加载文本文件
        
        参数:
            file_path: 文件路径
        
        返回:
            句子列表
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def __len__(self) -> int:
        """
        返回数据集大小
        """
        return len(self.src_sentences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        参数:
            idx: 样本索引
        
        返回:
            src_ids: 源序列ID [src_len]
            tgt_ids: 目标序列ID [tgt_len]
        """
        # 获取原始句子
        src_text = self.src_sentences[idx]
        tgt_text = self.tgt_sentences[idx]
        
        # 分词并编码
        src_ids = self.src_tokenizer.encode(src_text, add_bos=True, add_eos=True)
        tgt_ids = self.tgt_tokenizer.encode(tgt_text, add_bos=True, add_eos=True)
        
        # 截断到最大长度
        src_ids = src_ids[:self.max_length]
        tgt_ids = tgt_ids[:self.max_length]
        
        # 转换为张量
        src_tensor = torch.tensor(src_ids, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)
        
        return src_tensor, tgt_tensor


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_id: int = 0):
    """
    自定义批处理函数
    对不同长度的序列进行填充
    
    参数:
        batch: [(src_tensor, tgt_tensor), ...]
        pad_id: 填充token ID
    
    返回:
        src_batch: 填充后的源序列批次 [batch_size, max_src_len]
        tgt_batch: 填充后的目标序列批次 [batch_size, max_tgt_len]
    """
    # 分离源序列和目标序列
    src_batch, tgt_batch = zip(*batch)
    
    # 填充源序列到批次中的最大长度
    src_batch = torch.nn.utils.rnn.pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=pad_id
    )
    
    # 填充目标序列
    tgt_batch = torch.nn.utils.rnn.pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=pad_id
    )
    
    return src_batch, tgt_batch


class BucketBatchSampler:
    """
    按长度分桶的批采样器
    将相似长度的句子组成批次，提高效率
    
    参数:
        dataset: 数据集
        batch_size: 批大小
        shuffle: 是否打乱
    """
    
    def __init__(self, dataset: TranslationDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # 计算每个样本的长度 (源序列 + 目标序列)
        self.lengths = []
        for i in range(len(dataset)):
            src, tgt = dataset[i]
            self.lengths.append(len(src) + len(tgt))
        
        # 按长度排序的索引
        self.sorted_indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
    
    def __iter__(self):
        """
        生成批次
        """
        if self.shuffle:
            # 打乱但保持局部有序 (块内打乱)
            import random
            chunk_size = self.batch_size * 100
            indices = []
            for i in range(0, len(self.sorted_indices), chunk_size):
                chunk = self.sorted_indices[i:i + chunk_size]
                random.shuffle(chunk)
                indices.extend(chunk)
        else:
            indices = self.sorted_indices
        
        # 生成批次
        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        # 处理最后一个不完整的批次
        if batch:
            yield batch
    
    def __len__(self):
        """
        返回批次数量
        """
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
