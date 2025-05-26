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


def cal_iou(pred_bbox, gt_bbox):
    """
        计算iou
    """
    if pred_bbox is None or gt_bbox is None:
        return 0.0  # Return 0 IoU if either box is None
    # Determine the coordinates of the intersection rectangle
    x_left = max(pred_bbox[0], gt_bbox[0])
    y_top = max(pred_bbox[1], gt_bbox[1])
    x_right = min(pred_bbox[2], gt_bbox[2])
    y_bottom = min(pred_bbox[3], gt_bbox[3])

    # If there's no intersection, return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate areas of each bounding box
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])

    # Calculate union area
    union_area = pred_area + gt_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def check_bb(pred_bbox, gt_bbox):
    """
        判断预测的pred_bbox的中心点是否在gt_bbox里面
    """
    if pred_bbox is None or gt_bbox is None:
        # print("pred_bbox is None or gt_bbox is None")
        return False
    pred_center = [(pred_bbox[0] + pred_bbox[2]) / 2, (pred_bbox[1] + pred_bbox[3]) / 2]
    if pred_center[0] >= gt_bbox[0] and pred_center[0] <= gt_bbox[2] and pred_center[1] >= gt_bbox[1] and pred_center[
        1] <= gt_bbox[3]:
        return True
    else:
        return False


import re


def extract_bbox(s):
    """
        抽取x1, y1, x2, y2, 可以使用json进行解析，暂时使用正则问题也不大
    """
    pattern = r'"bbox_2d": *\[(\d+), *(\d+), *(\d+), *(\d+)\]'
    match = re.search(pattern, s)
    if match:
        return [float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))]
    return None


def compute_gr_iou_metrics(prediction) -> Dict[str, float]:
    """
        计算iou指标和bbox_acc指标
    """

    preds, labels = prediction[0], prediction[1]
    new_preds, new_labels = [], []
    for i in range(preds.shape[0]):
        # 序列化为可读文本
        new_preds.append(Serializer.from_tensor(preds[i]))
        new_labels.append(Serializer.from_tensor(labels[i]))

    correct_num, pre_num, all_iou = 0, len(new_preds), 0
    for pred, label in zip(new_preds, new_labels):
        # 对pred和gt进行bbox的正则抽取
        pred_bbox, label_bbox = extract_bbox(pred), extract_bbox(label)
        if check_bb(pred_bbox, label_bbox):
            correct_num += 1
        iou = cal_iou(pred_bbox, label_bbox)
        all_iou += iou
    avg_iou = all_iou / pre_num
    acc = correct_num / pre_num
    return {'bbox_acc': acc, 'iou': avg_iou}
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
    'iou': (compute_gr_iou_metrics, None),
}


def get_metric(metric: str):
    return METRIC_MAPPING[metric]
