"""
翻译器模块
提供简单易用的翻译接口
"""

import torch
from typing import List, Union
from .beam_search import BeamSearch


class Translator:
    """
    翻译器
    封装模型和分词器，提供简单的翻译接口
    
    参数:
        model: Transformer模型
        src_tokenizer: 源语言分词器
        tgt_tokenizer: 目标语言分词器
        device: 设备
        beam_size: beam搜索宽度（1表示贪心解码）
        max_length: 最大生成长度
    """
    
    def __init__(
        self,
        model,
        src_tokenizer,
        tgt_tokenizer,
        device='cuda',
        beam_size=1,
        max_length=512
    ):
        self.model = model.to(device)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        self.beam_size = beam_size
        self.max_length = max_length
        
        self.model.eval()
        
        # 特殊token ID
        self.bos_id = tgt_tokenizer.bos_id()
        self.eos_id = tgt_tokenizer.eos_id()
        self.pad_id = tgt_tokenizer.pad_id()
        
        # Beam Search解码器
        if beam_size > 1:
            self.beam_search = BeamSearch(
                model=model,
                beam_size=beam_size,
                max_length=max_length,
                device=device
            )
    
    def greedy_decode(self, src: torch.Tensor) -> List[int]:
        """
        贪心解码
        
        参数:
            src: 源序列 [1, src_len]
        
        返回:
            token ID列表
        """
        with torch.no_grad():
            # 编码
            encoder_output, src_mask = self.model.encode(src)
            
            # 初始化解码器输入
            tgt_input = torch.full(
                (1, 1),
                self.bos_id,
                dtype=torch.long,
                device=self.device
            )
            
            # 逐步生成
            for _ in range(self.max_length - 1):
                # 解码
                output = self.model.decode(tgt_input, encoder_output, src_mask)
                # [1, current_len, vocab_size]
                
                # 获取下一个token
                next_token = output[0, -1, :].argmax(dim=-1)
                
                # 检查是否结束
                if next_token.item() == self.eos_id:
                    tgt_input = torch.cat([
                        tgt_input,
                        next_token.unsqueeze(0).unsqueeze(0)
                    ], dim=1)
                    break
                
                # 拼接到输入
                tgt_input = torch.cat([
                    tgt_input,
                    next_token.unsqueeze(0).unsqueeze(0)
                ], dim=1)
            
            return tgt_input[0].tolist()
    
    def translate_sentence(
        self,
        sentence: str,
        return_score: bool = False
    ) -> Union[str, Tuple[str, float]]:
        """
        翻译单个句子
        
        参数:
            sentence: 源语言句子
            return_score: 是否返回分数
        
        返回:
            翻译结果（如果return_score=True，返回(translation, score)）
        """
        # 编码源句子
        src_ids = self.src_tokenizer.encode(sentence, add_bos=True, add_eos=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        
        # 解码
        if self.beam_size > 1:
            # Beam Search
            beams = self.beam_search.search(
                src_tensor,
                self.bos_id,
                self.eos_id,
                self.pad_id
            )
            tgt_ids, score = beams[0]
        else:
            # 贪心解码
            tgt_ids = self.greedy_decode(src_tensor)
            score = 0.0
        
        # 解码为文本
        translation = self.tgt_tokenizer.decode(tgt_ids)
        
        if return_score:
            return translation, score
        return translation
    
    def translate_batch(
        self,
        sentences: List[str],
        batch_size: int = 32
    ) -> List[str]:
        """
        批量翻译
        
        参数:
            sentences: 源语言句子列表
            batch_size: 批大小
        
        返回:
            翻译结果列表
        """
        translations = []
        
        # 分批处理
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            
            # 编码
            src_ids_list = [
                self.src_tokenizer.encode(sent, add_bos=True, add_eos=True)
                for sent in batch
            ]
            
            # 填充到相同长度
            max_len = max(len(ids) for ids in src_ids_list)
            src_batch = torch.zeros(
                len(batch),
                max_len,
                dtype=torch.long,
                device=self.device
            )
            
            for j, ids in enumerate(src_ids_list):
                src_batch[j, :len(ids)] = torch.tensor(ids)
            
            # 批量解码（目前使用循环，未来可优化为真正的批量推理）
            for src in src_batch:
                src_tensor = src.unsqueeze(0)  # [1, src_len]
                translation = self.translate_sentence(
                    self.src_tokenizer.decode(src.tolist())
                )
                translations.append(translation)
        
        return translations
    
    def interactive_translate(self):
        """
        交互式翻译
        """
        print("=" * 50)
        print("交互式翻译系统")
        print("=" * 50)
        print(f"Beam Size: {self.beam_size}")
        print(f"设备: {self.device}")
        print("输入'quit'或'exit'退出\n")
        
        while True:
            try:
                # 读取输入
                text = input("请输入要翻译的句子: ").strip()
                
                # 检查退出命令
                if text.lower() in ['quit', 'exit', 'q']:
                    print("再见!")
                    break
                
                if not text:
                    continue
                
                # 翻译
                if self.beam_size > 1:
                    translation, score = self.translate_sentence(text, return_score=True)
                    print(f"翻译结果: {translation}")
                    print(f"分数: {score:.4f}\n")
                else:
                    translation = self.translate_sentence(text)
                    print(f"翻译结果: {translation}\n")
                
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as e:
                print(f"错误: {e}\n")
