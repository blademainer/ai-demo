"""
数据准备脚本
下载、清洗和预处理翻译数据
训练SentencePiece分词器
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import sentencepiece as spm
from pathlib import Path
from src.data.preprocessor import TextPreprocessor


def train_sentencepiece(
    input_file,
    model_prefix,
    vocab_size=32000,
    model_type='bpe',
    character_coverage=0.9995
):
    """
    训练SentencePiece分词器
    
    参数:
        input_file: 训练文本文件
        model_prefix: 模型前缀
        vocab_size: 词汇表大小
        model_type: 模型类型 (bpe, unigram, char, word)
        character_coverage: 字符覆盖率
    """
    print(f"训练SentencePiece分词器: {model_prefix}")
    print(f"输入文件: {input_file}")
    print(f"词汇表大小: {vocab_size}")
    
    # 训练参数
    train_args = (
        f'--input={input_file} '
        f'--model_prefix={model_prefix} '
        f'--vocab_size={vocab_size} '
        f'--model_type={model_type} '
        f'--character_coverage={character_coverage} '
        f'--normalization_rule_name=nmt_nfkc '
        f'--split_by_whitespace=true '
        f'--byte_fallback=true '
        f'--unk_piece=<unk> '
        f'--bos_piece=<s> '
        f'--eos_piece=</s> '
        f'--pad_piece=<pad> '
        f'--unk_id=3 '
        f'--bos_id=1 '
        f'--eos_id=2 '
        f'--pad_id=0'
    )
    
    spm.SentencePieceTrainer.train(train_args)
    print(f"分词器训练完成: {model_prefix}.model")


def preprocess_data(args):
    """预处理数据"""
    print("开始数据预处理...")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建预处理器
    preprocessor = TextPreprocessor(
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    # 预处理训练数据
    if args.train_src and args.train_tgt:
        print("\n处理训练数据...")
        output_src = output_dir / 'train.zh'
        output_tgt = output_dir / 'train.en'
        
        count = preprocessor.preprocess_file(
            args.train_src,
            args.train_tgt,
            str(output_src),
            str(output_tgt)
        )
        print(f"训练数据: {count} 句对")
    
    # 预处理验证数据
    if args.val_src and args.val_tgt:
        print("\n处理验证数据...")
        output_src = output_dir / 'val.zh'
        output_tgt = output_dir / 'val.en'
        
        count = preprocessor.preprocess_file(
            args.val_src,
            args.val_tgt,
            str(output_src),
            str(output_tgt)
        )
        print(f"验证数据: {count} 句对")
    
    # 预处理测试数据
    if args.test_src and args.test_tgt:
        print("\n处理测试数据...")
        output_src = output_dir / 'test.zh'
        output_tgt = output_dir / 'test.en'
        
        count = preprocessor.preprocess_file(
            args.test_src,
            args.test_tgt,
            str(output_src),
            str(output_tgt)
        )
        print(f"测试数据: {count} 句对")
    
    print("\n数据预处理完成!")


def train_tokenizers(args):
    """训练分词器"""
    print("\n开始训练分词器...")
    
    # 创建词汇表目录
    vocab_dir = Path(args.vocab_dir)
    vocab_dir.mkdir(parents=True, exist_ok=True)
    
    processed_dir = Path(args.output_dir)
    
    # 训练源语言(中文)分词器
    src_train_file = processed_dir / 'train.zh'
    if src_train_file.exists():
        print("\n训练中文分词器...")
        train_sentencepiece(
            input_file=str(src_train_file),
            model_prefix=str(vocab_dir / 'spm_zh'),
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=0.9995
        )
    
    # 训练目标语言(英文)分词器
    tgt_train_file = processed_dir / 'train.en'
    if tgt_train_file.exists():
        print("\n训练英文分词器...")
        train_sentencepiece(
            input_file=str(tgt_train_file),
            model_prefix=str(vocab_dir / 'spm_en'),
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=1.0
        )
    
    print("\n分词器训练完成!")


def main():
    parser = argparse.ArgumentParser(description='数据准备和预处理')
    
    # 数据文件路径
    parser.add_argument('--train-src', type=str, help='训练集源语言文件')
    parser.add_argument('--train-tgt', type=str, help='训练集目标语言文件')
    parser.add_argument('--val-src', type=str, help='验证集源语言文件')
    parser.add_argument('--val-tgt', type=str, help='验证集目标语言文件')
    parser.add_argument('--test-src', type=str, help='测试集源语言文件')
    parser.add_argument('--test-tgt', type=str, help='测试集目标语言文件')
    
    # 输出路径
    parser.add_argument('--output-dir', type=str, default='./data/processed',
                        help='预处理后数据输出目录')
    parser.add_argument('--vocab-dir', type=str, default='./data/vocab',
                        help='分词器输出目录')
    
    # 预处理参数
    parser.add_argument('--min-length', type=int, default=5,
                        help='最小句子长度')
    parser.add_argument('--max-length', type=int, default=512,
                        help='最大句子长度')
    
    # 分词器参数
    parser.add_argument('--vocab-size', type=int, default=32000,
                        help='词汇表大小')
    parser.add_argument('--model-type', type=str, default='bpe',
                        choices=['bpe', 'unigram', 'char', 'word'],
                        help='分词模型类型')
    
    # 执行选项
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='跳过数据预处理')
    parser.add_argument('--skip-tokenizer', action='store_true',
                        help='跳过分词器训练')
    
    args = parser.parse_args()
    
    # 执行数据预处理
    if not args.skip_preprocess:
        preprocess_data(args)
    
    # 训练分词器
    if not args.skip_tokenizer:
        train_tokenizers(args)
    
    print("\n所有准备工作完成!")


if __name__ == '__main__':
    main()
