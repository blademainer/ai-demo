"""
训练脚本
使用配置文件训练Transformer翻译模型
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sentencepiece as spm
import yaml
import argparse
from pathlib import Path

from src.model import Transformer
from src.data import TranslationDataset, collate_fn
from src.training import Trainer, NoamLR, LabelSmoothingCrossEntropy


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


def create_model(model_config):
    """创建Transformer模型"""
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
    return model


def create_dataloader(data_config, src_tokenizer, tgt_tokenizer, split='train'):
    """创建数据加载器"""
    # 根据split选择数据文件
    if split == 'train':
        src_file = data_config['train_src']
        tgt_file = data_config['train_tgt']
        shuffle = True
    elif split == 'val':
        src_file = data_config['val_src']
        tgt_file = data_config['val_tgt']
        shuffle = False
    else:
        src_file = data_config['test_src']
        tgt_file = data_config['test_tgt']
        shuffle = False
    
    # 创建数据集
    dataset = TranslationDataset(
        src_file=src_file,
        tgt_file=tgt_file,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_length=data_config['max_length']
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=data_config.get('batch_size', 64),
        shuffle=shuffle,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        collate_fn=lambda batch: collate_fn(batch, pad_id=src_tokenizer.pad_id())
    )
    
    return dataloader


def main(args):
    """主训练函数"""
    # 加载配置
    print("加载配置文件...")
    model_config_path = args.model_config or './configs/model_config.yaml'
    train_config_path = args.train_config or './configs/training_config.yaml'
    
    model_cfg = load_config(model_config_path)['model']
    train_cfg = load_config(train_config_path)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() and train_cfg['device']['use_cuda'] else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载分词器
    print("加载分词器...")
    src_tokenizer = load_tokenizer(train_cfg['data']['src_tokenizer_path'])
    tgt_tokenizer = load_tokenizer(train_cfg['data']['tgt_tokenizer_path'])
    
    # 创建模型
    print("创建模型...")
    model = create_model(model_cfg)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 创建数据加载器
    print("加载数据...")
    train_dataloader = create_dataloader(
        {**train_cfg['data'], 'batch_size': train_cfg['training']['batch_size']},
        src_tokenizer,
        tgt_tokenizer,
        split='train'
    )
    
    val_dataloader = create_dataloader(
        {**train_cfg['data'], 'batch_size': train_cfg['training']['batch_size']},
        src_tokenizer,
        tgt_tokenizer,
        split='val'
    )
    
    print(f"训练集大小: {len(train_dataloader.dataset)}")
    print(f"验证集大小: {len(val_dataloader.dataset)}")
    
    # 创建损失函数
    criterion = LabelSmoothingCrossEntropy(
        vocab_size=model_cfg['tgt_vocab_size'],
        smoothing=train_cfg['training']['loss']['smoothing'],
        ignore_index=train_cfg['training']['loss']['ignore_index']
    )
    
    # 创建优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(
            train_cfg['training']['optimizer']['beta1'],
            train_cfg['training']['optimizer']['beta2']
        ),
        eps=train_cfg['training']['optimizer']['epsilon'],
        weight_decay=train_cfg['training']['optimizer']['weight_decay']
    )
    
    # 创建学习率调度器
    scheduler = NoamLR(
        optimizer,
        d_model=train_cfg['training']['scheduler']['d_model'],
        warmup_steps=train_cfg['training']['scheduler']['warmup_steps'],
        factor=train_cfg['training']['scheduler']['factor']
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config={
            'num_epochs': train_cfg['training']['num_epochs'],
            'gradient_accumulation_steps': train_cfg['training']['gradient_accumulation_steps'],
            'max_grad_norm': train_cfg['training']['max_grad_norm'],
            'checkpoint_dir': train_cfg['training']['checkpoint_dir'],
            'log_dir': train_cfg['training']['log_dir'],
            'save_interval': train_cfg['training']['save_interval'],
            'eval_interval': train_cfg['training']['eval_interval'],
            'log_interval': train_cfg['training']['log_interval'],
            'early_stopping_patience': train_cfg['training']['early_stopping_patience'],
            'fp16': train_cfg['training']['fp16']
        }
    )
    
    # 如果指定了检查点，加载检查点
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    print("\n" + "=" * 50)
    print("开始训练")
    print("=" * 50 + "\n")
    
    trainer.train()
    
    print("\n训练完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练Transformer翻译模型')
    parser.add_argument(
        '--model-config',
        type=str,
        default='./configs/model_config.yaml',
        help='模型配置文件路径'
    )
    parser.add_argument(
        '--train-config',
        type=str,
        default='./configs/training_config.yaml',
        help='训练配置文件路径'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='恢复训练的检查点路径'
    )
    
    args = parser.parse_args()
    main(args)
