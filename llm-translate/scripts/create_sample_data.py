"""
创建示例数据
用于快速测试和演示
"""

import os
from pathlib import Path


def create_sample_data():
    """创建示例中英文平行语料"""
    
    # 中英文示例句对
    zh_sentences = [
        "人工智能正在改变我们的生活方式。",
        "机器学习是人工智能的一个重要分支。",
        "深度学习在图像识别领域取得了巨大成功。",
        "自然语言处理使计算机能够理解人类语言。",
        "神经网络的灵感来自于人脑的工作方式。",
        "大数据为人工智能提供了丰富的训练资源。",
        "计算机视觉可以让机器看懂图像和视频。",
        "语音识别技术已经广泛应用于智能设备。",
        "推荐系统帮助用户发现感兴趣的内容。",
        "强化学习让机器通过试错来学习。",
        "云计算为AI模型提供了强大的算力支持。",
        "边缘计算可以降低数据传输的延迟。",
        "量子计算有望突破传统计算的瓶颈。",
        "区块链技术确保数据的安全和透明。",
        "5G网络将加速物联网的发展。",
        "虚拟现实创造了沉浸式的体验环境。",
        "增强现实将数字信息叠加到现实世界。",
        "自动驾驶技术正在重塑交通行业。",
        "智能家居让生活更加便捷舒适。",
        "医疗AI辅助医生进行疾病诊断。",
    ] * 50  # 复制50次，生成1000条训练数据
    
    en_sentences = [
        "Artificial intelligence is changing the way we live.",
        "Machine learning is an important branch of artificial intelligence.",
        "Deep learning has achieved great success in the field of image recognition.",
        "Natural language processing enables computers to understand human language.",
        "Neural networks are inspired by the way the human brain works.",
        "Big data provides rich training resources for artificial intelligence.",
        "Computer vision allows machines to understand images and videos.",
        "Speech recognition technology has been widely used in smart devices.",
        "Recommendation systems help users discover content of interest.",
        "Reinforcement learning allows machines to learn through trial and error.",
        "Cloud computing provides powerful computing support for AI models.",
        "Edge computing can reduce data transmission latency.",
        "Quantum computing is expected to break through the bottleneck of traditional computing.",
        "Blockchain technology ensures data security and transparency.",
        "5G networks will accelerate the development of the Internet of Things.",
        "Virtual reality creates an immersive experience environment.",
        "Augmented reality overlays digital information onto the real world.",
        "Autonomous driving technology is reshaping the transportation industry.",
        "Smart homes make life more convenient and comfortable.",
        "Medical AI assists doctors in disease diagnosis.",
    ] * 50
    
    # 创建数据目录
    data_dir = Path('./data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 划分数据集
    total = len(zh_sentences)
    train_size = int(total * 0.98)
    val_size = int(total * 0.01)
    
    train_zh = zh_sentences[:train_size]
    train_en = en_sentences[:train_size]
    
    val_zh = zh_sentences[train_size:train_size + val_size]
    val_en = en_sentences[train_size:train_size + val_size]
    
    test_zh = zh_sentences[train_size + val_size:]
    test_en = en_sentences[train_size + val_size:]
    
    # 保存训练集
    with open(data_dir / 'train.zh', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_zh))
    
    with open(data_dir / 'train.en', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_en))
    
    # 保存验证集
    with open(data_dir / 'val.zh', 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_zh))
    
    with open(data_dir / 'val.en', 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_en))
    
    # 保存测试集
    with open(data_dir / 'test.zh', 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_zh))
    
    with open(data_dir / 'test.en', 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_en))
    
    print(f"示例数据已创建在 {data_dir}")
    print(f"训练集: {len(train_zh)} 句对")
    print(f"验证集: {len(val_zh)} 句对")
    print(f"测试集: {len(test_zh)} 句对")
    print("\n下一步:")
    print("1. 运行数据预处理: python scripts/prepare_data.py \\")
    print("       --train-src data/raw/train.zh --train-tgt data/raw/train.en \\")
    print("       --val-src data/raw/val.zh --val-tgt data/raw/val.en \\")
    print("       --test-src data/raw/test.zh --test-tgt data/raw/test.en")
    print("\n2. 开始训练: python scripts/train.py")


if __name__ == '__main__':
    create_sample_data()
