"""
文本预处理模块
实现数据清洗、过滤和标准化
"""

import re
import unicodedata
from typing import List, Tuple


class TextPreprocessor:
    """
    文本预处理器
    执行以下操作:
    1. 去除HTML标签和特殊符号
    2. 统一标点符号
    3. 过滤异常长度句子
    4. Unicode标准化
    """
    
    def __init__(
        self,
        min_length=5,
        max_length=512,
        remove_html=True,
        normalize_punctuation=True
    ):
        """
        参数:
            min_length: 最小句子长度
            max_length: 最大句子长度
            remove_html: 是否去除HTML标签
            normalize_punctuation: 是否标准化标点符号
        """
        self.min_length = min_length
        self.max_length = max_length
        self.remove_html = remove_html
        self.normalize_punctuation = normalize_punctuation
        
        # 中英文标点符号映射
        self.punctuation_map = {
            '，': ',',
            '。': '.',
            '！': '!',
            '？': '?',
            '；': ';',
            '：': ':',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '（': '(',
            '）': ')',
            '【': '[',
            '】': ']',
            '《': '<',
            '》': '>',
        }
    
    def remove_html_tags(self, text: str) -> str:
        """
        去除HTML标签
        
        参数:
            text: 输入文本
        
        返回:
            清洗后的文本
        """
        # 去除HTML标签
        clean_text = re.sub(r'<[^>]+>', '', text)
        
        # 去除HTML实体
        clean_text = re.sub(r'&[a-zA-Z]+;', ' ', clean_text)
        
        return clean_text
    
    def normalize_unicode(self, text: str) -> str:
        """
        Unicode标准化
        
        参数:
            text: 输入文本
        
        返回:
            标准化后的文本
        """
        # NFKC标准化
        return unicodedata.normalize('NFKC', text)
    
    def normalize_punctuation_marks(self, text: str) -> str:
        """
        标准化标点符号 (中文标点转英文标点)
        
        参数:
            text: 输入文本
        
        返回:
            标准化后的文本
        """
        for zh_punct, en_punct in self.punctuation_map.items():
            text = text.replace(zh_punct, en_punct)
        return text
    
    def remove_extra_spaces(self, text: str) -> str:
        """
        去除多余空格
        
        参数:
            text: 输入文本
        
        返回:
            清洗后的文本
        """
        # 将多个空格替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        
        # 去除首尾空格
        return text.strip()
    
    def is_valid_length(self, text: str) -> bool:
        """
        检查文本长度是否有效
        
        参数:
            text: 输入文本
        
        返回:
            是否有效
        """
        length = len(text.split())
        return self.min_length <= length <= self.max_length
    
    def preprocess(self, text: str) -> str:
        """
        执行完整的预处理流程
        
        参数:
            text: 输入文本
        
        返回:
            预处理后的文本
        """
        # 去除HTML标签
        if self.remove_html:
            text = self.remove_html_tags(text)
        
        # Unicode标准化
        text = self.normalize_unicode(text)
        
        # 标准化标点符号
        if self.normalize_punctuation:
            text = self.normalize_punctuation_marks(text)
        
        # 去除多余空格
        text = self.remove_extra_spaces(text)
        
        return text
    
    def preprocess_pair(self, src_text: str, tgt_text: str) -> Tuple[str, str]:
        """
        预处理句对
        
        参数:
            src_text: 源语言文本
            tgt_text: 目标语言文本
        
        返回:
            预处理后的句对，如果无效则返回None
        """
        # 预处理
        src_clean = self.preprocess(src_text)
        tgt_clean = self.preprocess(tgt_text)
        
        # 验证长度
        if not (self.is_valid_length(src_clean) and self.is_valid_length(tgt_clean)):
            return None
        
        return src_clean, tgt_clean
    
    def preprocess_file(
        self,
        src_file: str,
        tgt_file: str,
        output_src_file: str,
        output_tgt_file: str
    ) -> int:
        """
        预处理平行语料文件
        
        参数:
            src_file: 源语言文件路径
            tgt_file: 目标语言文件路径
            output_src_file: 输出源语言文件路径
            output_tgt_file: 输出目标语言文件路径
        
        返回:
            处理后的句对数量
        """
        valid_pairs = 0
        
        with open(src_file, 'r', encoding='utf-8') as f_src, \
             open(tgt_file, 'r', encoding='utf-8') as f_tgt, \
             open(output_src_file, 'w', encoding='utf-8') as f_out_src, \
             open(output_tgt_file, 'w', encoding='utf-8') as f_out_tgt:
            
            for src_line, tgt_line in zip(f_src, f_tgt):
                # 预处理句对
                result = self.preprocess_pair(src_line.strip(), tgt_line.strip())
                
                if result is not None:
                    src_clean, tgt_clean = result
                    f_out_src.write(src_clean + '\n')
                    f_out_tgt.write(tgt_clean + '\n')
                    valid_pairs += 1
        
        return valid_pairs
