"""
评估指标模块
实现BLEU等翻译质量评估指标
"""

from typing import List, Dict
import sacrebleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# 下载NLTK数据 (首次运行时需要)
try:
    nltk.data.find('wordnet')
except LookupError:
    print("正在下载NLTK数据...")
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


def compute_bleu(
    predictions: List[str],
    references: List[List[str]],
    smooth: bool = True
) -> Dict[str, float]:
    """
    计算BLEU分数
    
    参数:
        predictions: 预测句子列表
        references: 参考句子列表的列表 (每个预测可以有多个参考)
        smooth: 是否使用平滑
    
    返回:
        BLEU分数字典 (包含BLEU-1, BLEU-2, BLEU-3, BLEU-4)
    """
    # 使用sacrebleu计算corpus-level BLEU
    # sacrebleu需要references为List[List[str]]格式
    # 其中外层list是不同的参考翻译，内层list是句子
    
    # 转换格式: [[ref1_sent1, ref1_sent2, ...], [ref2_sent1, ref2_sent2, ...]]
    if references and isinstance(references[0], list):
        # 已经是正确格式
        refs_transposed = list(zip(*references))
        refs_for_bleu = [[ref] for ref in refs_transposed[0]] if refs_transposed else [[]]
    else:
        # 每个预测只有一个参考
        refs_for_bleu = [[ref] for ref in references]
    
    # 计算corpus-level BLEU-4
    bleu = sacrebleu.corpus_bleu(predictions, refs_for_bleu)
    
    results = {
        'bleu': bleu.score,
        'bleu_1': bleu.precisions[0],
        'bleu_2': bleu.precisions[1],
        'bleu_3': bleu.precisions[2],
        'bleu_4': bleu.precisions[3],
    }
    
    return results


def compute_sentence_bleu(
    prediction: str,
    references: List[str],
    weights: tuple = (0.25, 0.25, 0.25, 0.25)
) -> float:
    """
    计算单句BLEU分数
    
    参数:
        prediction: 预测句子
        references: 参考句子列表
        weights: n-gram权重
    
    返回:
        BLEU分数
    """
    # 分词
    pred_tokens = prediction.split()
    ref_tokens_list = [ref.split() for ref in references]
    
    # 使用平滑函数
    smooth_fn = SmoothingFunction().method1
    
    bleu = sentence_bleu(
        ref_tokens_list,
        pred_tokens,
        weights=weights,
        smoothing_function=smooth_fn
    )
    
    return bleu


def compute_meteor(
    predictions: List[str],
    references: List[str]
) -> float:
    """
    计算METEOR分数
    
    参数:
        predictions: 预测句子列表
        references: 参考句子列表
    
    返回:
        平均METEOR分数
    """
    scores = []
    
    for pred, ref in zip(predictions, references):
        # 分词
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        # 计算METEOR
        score = meteor_score([ref_tokens], pred_tokens)
        scores.append(score)
    
    return sum(scores) / len(scores) if scores else 0.0


def compute_chrf(
    predictions: List[str],
    references: List[List[str]]
) -> float:
    """
    计算chrF分数 (字符级F-score)
    
    参数:
        predictions: 预测句子列表
        references: 参考句子列表的列表
    
    返回:
        chrF分数
    """
    # 转换格式
    if references and isinstance(references[0], list):
        refs_transposed = list(zip(*references))
        refs_for_chrf = [[ref] for ref in refs_transposed[0]] if refs_transposed else [[]]
    else:
        refs_for_chrf = [[ref] for ref in references]
    
    chrf = sacrebleu.corpus_chrf(predictions, refs_for_chrf)
    
    return chrf.score


def compute_metrics(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    计算所有评估指标
    
    参数:
        predictions: 预测句子列表
        references: 参考句子列表
    
    返回:
        指标字典
    """
    metrics = {}
    
    # BLEU
    bleu_results = compute_bleu(predictions, [[ref] for ref in references])
    metrics.update(bleu_results)
    
    # chrF
    try:
        chrf_score = compute_chrf(predictions, [[ref] for ref in references])
        metrics['chrf'] = chrf_score
    except Exception as e:
        print(f"计算chrF时出错: {e}")
        metrics['chrf'] = 0.0
    
    # METEOR (可选，计算较慢)
    try:
        meteor = compute_meteor(predictions, references)
        metrics['meteor'] = meteor
    except Exception as e:
        print(f"计算METEOR时出错: {e}")
        metrics['meteor'] = 0.0
    
    return metrics


def print_metrics(metrics: Dict[str, float]):
    """
    打印评估指标
    
    参数:
        metrics: 指标字典
    """
    print("\n" + "=" * 50)
    print("评估结果:")
    print("=" * 50)
    
    for key, value in metrics.items():
        if 'bleu' in key.lower():
            print(f"{key.upper():15s}: {value:6.2f}")
        else:
            print(f"{key.upper():15s}: {value:6.4f}")
    
    print("=" * 50 + "\n")
