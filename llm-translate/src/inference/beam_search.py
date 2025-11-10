"""
Beam Search解码模块
实现束搜索算法用于高质量翻译
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


class BeamSearch:
    """
    Beam Search解码器
    
    参数:
        model: Transformer模型
        beam_size: beam宽度 (默认5)
        max_length: 最大生成长度 (默认512)
        length_penalty: 长度惩罚 (默认0.6)
        no_repeat_ngram_size: 防止n-gram重复 (默认3)
        device: 设备
    """
    
    def __init__(
        self,
        model,
        beam_size=5,
        max_length=512,
        length_penalty=0.6,
        no_repeat_ngram_size=3,
        device='cuda'
    ):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.device = device
        
        self.model.eval()
    
    def length_penalty_fn(self, length):
        """
        长度惩罚函数
        公式: ((5 + length) / 6) ^ alpha
        
        参数:
            length: 序列长度
        
        返回:
            惩罚系数
        """
        return ((5 + length) / 6) ** self.length_penalty
    
    def get_ngrams(self, tokens, n):
        """
        获取n-gram集合
        
        参数:
            tokens: token序列
            n: n-gram大小
        
        返回:
            n-gram集合
        """
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.add(ngram)
        return ngrams
    
    def should_block_ngram(self, current_tokens, next_token):
        """
        检查是否应该阻止该token（防止n-gram重复）
        
        参数:
            current_tokens: 当前token序列
            next_token: 候选下一个token
        
        返回:
            是否应该阻止
        """
        if self.no_repeat_ngram_size == 0:
            return False
        
        # 检查添加next_token后是否产生重复的n-gram
        new_tokens = current_tokens + [next_token]
        
        if len(new_tokens) < self.no_repeat_ngram_size:
            return False
        
        # 获取最新的n-gram
        new_ngram = tuple(new_tokens[-self.no_repeat_ngram_size:])
        
        # 检查是否在之前的序列中出现过
        for i in range(len(new_tokens) - self.no_repeat_ngram_size):
            if tuple(new_tokens[i:i + self.no_repeat_ngram_size]) == new_ngram:
                return True
        
        return False
    
    def search(
        self,
        src: torch.Tensor,
        bos_id: int,
        eos_id: int,
        pad_id: int
    ) -> List[Tuple[List[int], float]]:
        """
        执行Beam Search
        
        参数:
            src: 源序列 [1, src_len]
            bos_id: 句子开始token ID
            eos_id: 句子结束token ID
            pad_id: 填充token ID
        
        返回:
            [(token_ids, score), ...] 按分数排序的候选序列列表
        """
        batch_size = src.size(0)
        assert batch_size == 1, "Beam search目前仅支持batch_size=1"
        
        with torch.no_grad():
            # 编码源序列
            encoder_output, src_mask = self.model.encode(src)
            # [1, src_len, d_model]
            
            # 初始化beam
            # beams: [(tokens, score), ...]
            beams = [([bos_id], 0.0)]
            completed_beams = []
            
            # 逐步生成
            for step in range(self.max_length):
                candidates = []
                
                for tokens, score in beams:
                    # 如果已经结束，加入完成列表
                    if tokens[-1] == eos_id:
                        completed_beams.append((tokens, score))
                        continue
                    
                    # 准备解码器输入
                    tgt_input = torch.tensor([tokens], dtype=torch.long, device=self.device)
                    # [1, current_len]
                    
                    # 解码
                    output = self.model.decode(tgt_input, encoder_output, src_mask)
                    # [1, current_len, vocab_size]
                    
                    # 获取最后一个位置的logits
                    next_token_logits = output[0, -1, :]  # [vocab_size]
                    
                    # 计算log概率
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                    
                    # 获取top-k候选
                    top_log_probs, top_indices = torch.topk(log_probs, self.beam_size * 2)
                    
                    # 遍历候选
                    for log_prob, token_id in zip(top_log_probs, top_indices):
                        token_id = token_id.item()
                        
                        # 检查n-gram重复
                        if self.should_block_ngram(tokens, token_id):
                            continue
                        
                        # 计算新分数
                        new_score = score + log_prob.item()
                        new_tokens = tokens + [token_id]
                        
                        candidates.append((new_tokens, new_score))
                
                # 如果没有候选，退出
                if not candidates:
                    break
                
                # 按分数排序并选择top beam_size
                candidates.sort(key=lambda x: x[1] / self.length_penalty_fn(len(x[0])), reverse=True)
                beams = candidates[:self.beam_size]
                
                # 如果所有beam都完成，退出
                if len(completed_beams) >= self.beam_size:
                    break
            
            # 合并完成和未完成的beam
            all_beams = completed_beams + beams
            
            # 应用长度惩罚并排序
            all_beams.sort(
                key=lambda x: x[1] / self.length_penalty_fn(len(x[0])),
                reverse=True
            )
            
            return all_beams[:self.beam_size]
