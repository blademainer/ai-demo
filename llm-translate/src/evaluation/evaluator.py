"""
评估器模块
实现模型评估流程
"""

import torch
from torch.utils.data import DataLoader
from typing import List, Dict
from tqdm import tqdm
from .metrics import compute_metrics, print_metrics


class Evaluator:
    """
    模型评估器
    
    参数:
        model: 翻译模型
        tokenizer: 分词器
        device: 评估设备
    """
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def translate_batch(
        self,
        src_batch: torch.Tensor,
        max_length: int = 512,
        beam_size: int = 1
    ) -> List[List[int]]:
        """
        批量翻译 (贪心解码)
        
        参数:
            src_batch: 源序列批次 [batch_size, src_len]
            max_length: 最大生成长度
            beam_size: beam搜索宽度 (当前仅支持贪心解码，beam_size=1)
        
        返回:
            翻译结果ID列表
        """
        batch_size = src_batch.size(0)
        
        # 编码源序列
        with torch.no_grad():
            encoder_output, src_mask = self.model.encode(src_batch)
        
        # 初始化解码器输入 (所有句子以<bos>开始)
        bos_id = self.tokenizer.bos_id()
        eos_id = self.tokenizer.eos_id()
        
        # [batch_size, 1]
        tgt_input = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=self.device)
        
        # 存储每个句子是否已完成
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # 逐步生成
        for _ in range(max_length - 1):
            with torch.no_grad():
                # 解码
                output = self.model.decode(tgt_input, encoder_output, src_mask)
                # [batch_size, current_len, vocab_size]
                
                # 获取最后一个位置的预测
                next_token_logits = output[:, -1, :]  # [batch_size, vocab_size]
                
                # 贪心选择
                next_tokens = torch.argmax(next_token_logits, dim=-1)  # [batch_size]
                
                # 标记已完成的句子
                finished = finished | (next_tokens == eos_id)
                
                # 将<eos>后的token设为<eos> (保持已完成句子不变)
                next_tokens = torch.where(finished, torch.tensor(eos_id, device=self.device), next_tokens)
                
                # 拼接到输入序列
                tgt_input = torch.cat([tgt_input, next_tokens.unsqueeze(1)], dim=1)
                
                # 如果所有句子都已完成，提前退出
                if finished.all():
                    break
        
        # 转换为列表
        translations = tgt_input.cpu().tolist()
        
        return translations
    
    def evaluate(
        self,
        dataloader: DataLoader,
        max_samples: int = None
    ) -> Dict[str, float]:
        """
        评估模型
        
        参数:
            dataloader: 测试数据加载器
            max_samples: 最大评估样本数 (None表示全部)
        
        返回:
            评估指标字典
        """
        predictions = []
        references = []
        
        sample_count = 0
        
        print("开始评估...")
        
        for src_batch, tgt_batch in tqdm(dataloader, desc='Evaluating'):
            # 移动到设备
            src_batch = src_batch.to(self.device)
            
            # 翻译
            pred_ids_batch = self.translate_batch(src_batch)
            
            # 解码为文本
            for pred_ids, tgt_ids in zip(pred_ids_batch, tgt_batch.tolist()):
                # 去除特殊token并解码
                pred_text = self.tokenizer.decode(pred_ids)
                ref_text = self.tokenizer.decode(tgt_ids)
                
                predictions.append(pred_text)
                references.append(ref_text)
                
                sample_count += 1
                
                # 打印一些样本
                if sample_count <= 5:
                    print(f"\n样本 {sample_count}:")
                    print(f"预测: {pred_text}")
                    print(f"参考: {ref_text}")
                
                # 检查是否达到最大样本数
                if max_samples and sample_count >= max_samples:
                    break
            
            if max_samples and sample_count >= max_samples:
                break
        
        # 计算指标
        metrics = compute_metrics(predictions, references)
        
        # 打印结果
        print_metrics(metrics)
        
        return metrics
    
    def translate_sentences(
        self,
        sentences: List[str],
        max_length: int = 512
    ) -> List[str]:
        """
        翻译句子列表
        
        参数:
            sentences: 源语言句子列表
            max_length: 最大生成长度
        
        返回:
            翻译结果列表
        """
        translations = []
        
        for sentence in tqdm(sentences, desc='Translating'):
            # 编码
            src_ids = self.tokenizer.encode(sentence, add_bos=True, add_eos=True)
            src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)
            
            # 翻译
            pred_ids = self.translate_batch(src_tensor, max_length)[0]
            
            # 解码
            translation = self.tokenizer.decode(pred_ids)
            translations.append(translation)
        
        return translations
