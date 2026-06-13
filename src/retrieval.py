"""图文检索评测的通用工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass(frozen=True)
class RetrievalItem:
    """图文检索样本。

    Args:
        image_id: 图像唯一标识。
        image_path: 图像文件路径。
        captions: 与图像对应的文本描述列表。
    """

    image_id: str
    image_path: Path
    captions: list[str]


def read_items(path: Path, max_images: int = 0) -> list[RetrievalItem]:
    """读取 JSONL 格式的图文检索标注。

    Args:
        path: 标注文件路径。
        max_images: 最多读取的图像数量，0 表示读取全部。

    Returns:
        检索样本列表。

    Raises:
        ValueError: 标注文件中没有有效样本。
        FileNotFoundError: 图像文件不存在。
    """

    items: list[RetrievalItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            captions = [str(caption).strip() for caption in row["captions"]]
            captions = [caption for caption in captions if caption]
            if not captions:
                continue
            items.append(
                RetrievalItem(
                    image_id=str(row["image_id"]),
                    image_path=Path(row["image_path"]),
                    captions=captions,
                )
            )
            if max_images > 0 and len(items) >= max_images:
                break

    if not items:
        raise ValueError(f"没有从 {path} 读取到有效样本。")

    missing = [str(item.image_path) for item in items if not item.image_path.exists()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"有 {len(missing)} 张图像不存在，前 10 个为:\n{preview}")

    return items


def flatten_captions(items: Iterable[RetrievalItem]) -> tuple[list[str], list[int]]:
    """展开 caption 并记录每张图像对应的 caption 数。

    Args:
        items: 检索样本。

    Returns:
        文本列表和每张图像的文本数量列表。
    """

    texts: list[str] = []
    counts: list[int] = []
    for item in items:
        texts.extend(item.captions)
        counts.append(len(item.captions))
    return texts, counts


def normalize(features: torch.Tensor) -> torch.Tensor:
    """对特征做 L2 归一化。

    Args:
        features: 特征矩阵。

    Returns:
        归一化后的特征矩阵。
    """

    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def recall_at(ranks: list[int], k_value: int) -> float:
    """计算 Recall@K 百分比。

    Args:
        ranks: 正确候选的排序位置，0 表示第一名。
        k_value: Recall@K 中的 K。

    Returns:
        百分比形式的召回率。
    """

    return 100.0 * sum(rank < k_value for rank in ranks) / len(ranks)


def compute_retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    caption_counts: list[int],
) -> dict[str, float]:
    """计算图到文和文到图检索指标。

    Args:
        image_features: 图像特征矩阵。
        text_features: 文本特征矩阵。
        caption_counts: 每张图像对应的文本数量。

    Returns:
        包含 R@1、R@5、R@10 和 RSum 的指标字典。
    """

    similarity = normalize(image_features.float()) @ normalize(text_features.float()).T
    caption_offsets: list[int] = []
    offset = 0
    for count in caption_counts:
        caption_offsets.append(offset)
        offset += count

    image_to_text_ranks: list[int] = []
    sorted_text_indices = torch.argsort(similarity, dim=1, descending=True)
    for image_index, caption_count in enumerate(caption_counts):
        target_start = caption_offsets[image_index]
        targets = set(range(target_start, target_start + caption_count))
        ranked = sorted_text_indices[image_index].tolist()
        image_to_text_ranks.append(min(ranked.index(index) for index in targets))

    caption_to_image: list[int] = []
    for image_index, caption_count in enumerate(caption_counts):
        caption_to_image.extend([image_index] * caption_count)

    text_to_image_ranks: list[int] = []
    sorted_image_indices = torch.argsort(similarity, dim=0, descending=True)
    for caption_index, target_image in enumerate(caption_to_image):
        ranked = sorted_image_indices[:, caption_index].tolist()
        text_to_image_ranks.append(ranked.index(target_image))

    metrics = {
        "image_to_text_r1": recall_at(image_to_text_ranks, 1),
        "image_to_text_r5": recall_at(image_to_text_ranks, 5),
        "image_to_text_r10": recall_at(image_to_text_ranks, 10),
        "text_to_image_r1": recall_at(text_to_image_ranks, 1),
        "text_to_image_r5": recall_at(text_to_image_ranks, 5),
        "text_to_image_r10": recall_at(text_to_image_ranks, 10),
    }
    metrics["rsum"] = sum(metrics.values())
    return metrics
