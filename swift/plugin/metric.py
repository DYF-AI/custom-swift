# Copyright (c) Alibaba, Inc. and its affiliates.
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Literal

import numpy as np
import torch
from transformers.trainer_utils import EvalPrediction

from swift.utils import Serializer, get_logger

logger = get_logger()


class Metric(ABC):

    def __init__(self):
        self._default = {}
        self._default_factory = {}

    def add_state(self, name: str, default=None, default_factory=None) -> None:
        if not hasattr(self, '_default'):
            raise AttributeError('Please call super().__init__() first.')
        if default is None:
            self._default_factory[name] = default_factory
            assert name not in self._default, f'self._default: {self._default}'
            default = default_factory()
        else:
            self._default[name] = default
            assert name not in self._default_factory, f'self._default_factory: {self._default_factory}'
        setattr(self, name, default)

    def reset(self):
        for k, v in self._default.items():
            setattr(self, k, v)
        for k, v in self._default_factory.items():
            setattr(self, k, v())

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute(self):
        pass


class InferStats(Metric):

    def __init__(self):
        super().__init__()
        self.add_state('start_runtime', default_factory=lambda: time.perf_counter())
        self.add_state('num_prompt_tokens', default_factory=dict)
        self.add_state('num_generated_tokens', default_factory=dict)

    def update(self, output):
        id_ = output.id
        self.num_prompt_tokens[id_] = output.usage.prompt_tokens
        self.num_generated_tokens[id_] = output.usage.completion_tokens

    def compute(self):
        runtime = time.perf_counter() - self.start_runtime
        num_samples = len(self.num_generated_tokens)
        num_generated_tokens = sum(self.num_generated_tokens.values())
        return {
            'num_prompt_tokens': sum(self.num_prompt_tokens.values()),
            'num_generated_tokens': num_generated_tokens,
            'num_samples': num_samples,
            'runtime': runtime,
            'samples/s': num_samples / runtime,
            'tokens/s': num_generated_tokens / runtime,
        }


class MeanMetric(Metric):

    def __init__(self, nan_value=0):
        super().__init__()
        self.nan_value = nan_value
        self.add_state('state', default=0.)
        self.add_state('count', default=0)

    def update(self, state: torch.Tensor):
        if isinstance(state, (torch.Tensor, np.ndarray)):
            state = state.tolist()

        if isinstance(state, (list, tuple)):
            count = len(state)
            state = sum(state)
        else:
            count = 1

        self.state += state
        self.count += count

    def compute(self):
        return {
            'value': self.state / self.count if self.count > 0 else self.nan_value,
        }


def compute_rouge_bleu(preds: List[str], labels: List[str]):
    import jieba
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge.rouge import Rouge
    score_dict = {key: MeanMetric() for key in ['rouge-1', 'rouge-2', 'rouge-l', 'bleu-4']}

    for pred, label in zip(preds, labels):
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        if not hypothesis or not reference:
            continue
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))[0]
        for k, v in scores.items():
            score_dict[k].update(v['f'])
        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict['bleu-4'].update(bleu_score)

    return {k: round(v.compute()['value'] * 100, 6) for k, v in score_dict.items()}


def compute_nlg_metrics(prediction) -> Dict[str, float]:
    preds, labels = prediction[0], prediction[1]
    new_preds, new_labels = [], []
    for i in range(preds.shape[0]):
        new_preds.append(Serializer.from_tensor(preds[i]))
        new_labels.append(Serializer.from_tensor(labels[i]))
    return compute_rouge_bleu(new_preds, new_labels)


def calculate_iou(boxA, boxB):
    """
    计算两个边界框的 IoU。
    :param boxA: 第一个边界框 [x1, y1, x2, y2]
    :param boxB: 第二个边界框 [x1, y1, x2, y2]
    :return: IoU 值
    """
    # 计算交集区域的坐标
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 计算交集区域的宽和高
    inter_width = xB - xA
    inter_height = yB - yA

    # 处理无交集情况
    if inter_width <= 0 or inter_height <= 0:
        return 0.0

    # 计算交集和并集面积
    inter_area = inter_width * inter_height
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area

    # 计算 IoU
    iou = inter_area / union_area
    return iou

import re
import json


def compute_iou(new_preds, new_labels):
    """
    支持容错的 IoU 计算：用正则提取 JSON 结构，解析失败时返回 0
    :param new_preds: 预测框的字符串列表，如 ['```json[{"bbox_2d": [...]}]```', ...]
    :param new_labels: 标签框的字符串列表
    :return: (iou_list, avg_iou)
    """
    # 正则提取 ```json 包裹的 JSON 内容（兼容多余字符）
    json_pattern = re.compile(r"```json(.*?)```", re.DOTALL)
    iou_list = []

    for pred_str, label_str in zip(new_preds, new_labels):
        try:
            # 提取预测框 JSON 部分
            pred_json_str = json_pattern.search(pred_str).group(1).strip()
            pred_data = json.loads(pred_json_str)
            pred_box = pred_data[0]["bbox_2d"]  # 假设每个元素是列表且第一个对象含 bbox_2d

            # 提取标签框 JSON 部分
            label_json_str = json_pattern.search(label_str).group(1).strip()
            label_data = json.loads(label_json_str)
            label_box = label_data[0]["bbox_2d"]

            # 计算 IoU
            iou = calculate_iou(pred_box, label_box)
            iou_list.append(iou)
        except (AttributeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            # 捕获所有可能的解析错误（正则未匹配、JSON 格式错误、键缺失、索引越界、类型错误）
            iou_list.append(0.0)

    avg_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0
    return iou_list, avg_iou


def compute_iou_metrics(prediction) -> Dict[str, float]:
    preds, labels = prediction[0], prediction[1]
    new_preds, new_labels = [], []
    for i in range(preds.shape[0]):
        new_preds.append(Serializer.from_tensor(preds[i]))
        new_labels.append(Serializer.from_tensor(labels[i]))
    return compute_iou(new_preds, new_labels)

def compute_acc(preds,
                labels,
                *,
                acc_strategy: Literal['token', 'seq'] = 'token',
                is_encoder_decoder: bool = False) -> Dict[str, List[float]]:

    if isinstance(preds, torch.Tensor):
        if torch.is_floating_point(labels):
            return {}
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
    if preds.ndim >= 2 and not is_encoder_decoder:
        labels = labels[..., 1:]
        preds = preds[..., :-1]
    if np.issubdtype(labels.dtype, np.floating) or preds.shape != labels.shape:
        return {}

    masks = labels != -100
    if acc_strategy == 'token' or preds.ndim == 1:
        acc_list = (preds[masks] == labels[masks]).tolist()
    else:
        acc_list = []
        for i, m in enumerate(masks):
            acc_list.append(np.all(preds[i, m] == labels[i, m]))
    return {f'{acc_strategy}_acc' if preds.ndim >= 2 else 'acc': acc_list}


def compute_acc_metrics(eval_prediction: EvalPrediction,
                        *,
                        acc_strategy: Literal['token', 'seq'] = 'token',
                        is_encoder_decoder: bool = False) -> Dict[str, float]:

    metric = compute_acc(
        eval_prediction.predictions,
        eval_prediction.label_ids,
        acc_strategy=acc_strategy,
        is_encoder_decoder=is_encoder_decoder)
    if len(metric) == 0:
        return {}
    return {k: sum(v) / len(v) for k, v in metric.items()}


def preprocess_logits_for_acc(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    preds = logits.argmax(dim=-1)
    return preds


# Add your own metric calculation method here, use --metric xxx to train
METRIC_MAPPING = {
    'acc': (compute_acc_metrics, preprocess_logits_for_acc),
    'nlg': (compute_nlg_metrics, None),
}


def get_metric(metric: str):
    return METRIC_MAPPING[metric]
