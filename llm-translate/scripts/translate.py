"""
交互式翻译脚本
加载训练好的模型进行翻译
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import sentencepiece as spm
import yaml
import argparse

from src.model import Transformer
from src.inference import Translator


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_tokenizer(tokenizer_path):
    """加载SentencePiece分词器"""
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    return sp


def load_model(checkpoint_path, model_config, device):
    """加载训练好的模型"""
    model = Transformer(
        src_vocab_size=model_config['src_vocab_size'],
        tgt_vocab_size=model_config['tgt_vocab_size'],
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        num_encoder_layers=model_config['num_encoder_layers'],
        num_decoder_layers=model_config['num_decoder_layers'],
        d_ff=model_config['d_ff'],
        max_seq_length=model_config['max_seq_length'],
        dropout=model_config['dropout'],
        pad_idx=model_config['pad_idx']
    )
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ 加载模型: {checkpoint_path}")
    
    return model


def main(args):
    """主函数"""
    # 加载配置
    print("加载配置...")
    model_config_path = args.model_config or './configs/model_config.yaml'
    train_config_path = args.train_config or './configs/training_config.yaml'
    
    model_cfg = load_config(model_config_path)['model']
    train_cfg = load_config(train_config_path)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"✓ 使用设备: {device}")
    
    # 加载分词器
    print("加载分词器...")
    src_tokenizer = load_tokenizer(train_cfg['data']['src_tokenizer_path'])
    tgt_tokenizer = load_tokenizer(train_cfg['data']['tgt_tokenizer_path'])
    print(f"✓ 源语言词汇表: {src_tokenizer.vocab_size()}")
    print(f"✓ 目标语言词汇表: {tgt_tokenizer.vocab_size()}")
    
    # 加载模型
    print("加载模型...")
    model = load_model(args.checkpoint, model_cfg, device)
    
    # 创建翻译器
    translator = Translator(
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        device=device,
        beam_size=args.beam_size,
        max_length=args.max_length
    )
    
    # 根据模式执行
    if args.mode == 'interactive':
        # 交互式翻译
        translator.interactive_translate()
    
    elif args.mode == 'file':
        # 文件翻译
        if not args.input_file:
            print("错误: 文件模式需要指定--input-file")
            return
        
        print(f"\n从文件翻译: {args.input_file}")
        
        # 读取输入文件
        with open(args.input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        print(f"共 {len(sentences)} 个句子")
        
        # 翻译
        translations = []
        for i, sent in enumerate(sentences, 1):
            print(f"\r翻译进度: {i}/{len(sentences)}", end='', flush=True)
            translation = translator.translate_sentence(sent)
            translations.append(translation)
        
        print("\n翻译完成!")
        
        # 保存结果
        output_file = args.output_file or args.input_file + '.trans'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(translations))
        
        print(f"结果已保存到: {output_file}")
    
    elif args.mode == 'single':
        # 单句翻译
        if not args.text:
            print("错误: 单句模式需要指定--text")
            return
        
        print(f"\n原文: {args.text}")
        
        if args.beam_size > 1:
            translation, score = translator.translate_sentence(args.text, return_score=True)
            print(f"译文: {translation}")
            print(f"分数: {score:.4f}")
        else:
            translation = translator.translate_sentence(args.text)
            print(f"译文: {translation}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer翻译系统')
    
    # 模型和配置
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='模型检查点路径'
    )
    parser.add_argument(
        '--model-config',
        type=str,
        default='./configs/model_config.yaml',
        help='模型配置文件'
    )
    parser.add_argument(
        '--train-config',
        type=str,
        default='./configs/training_config.yaml',
        help='训练配置文件'
    )
    
    # 翻译参数
    parser.add_argument(
        '--beam-size',
        type=int,
        default=1,
        help='Beam搜索宽度 (1=贪心解码)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='最大生成长度'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='使用CPU而不是GPU'
    )
    
    # 模式选择
    parser.add_argument(
        '--mode',
        type=str,
        default='interactive',
        choices=['interactive', 'file', 'single'],
        help='翻译模式: interactive(交互), file(文件), single(单句)'
    )
    
    # 文件模式参数
    parser.add_argument(
        '--input-file',
        type=str,
        help='输入文件路径 (文件模式)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='输出文件路径 (文件模式)'
    )
    
    # 单句模式参数
    parser.add_argument(
        '--text',
        type=str,
        help='要翻译的文本 (单句模式)'
    )
    
    args = parser.parse_args()
    main(args)
