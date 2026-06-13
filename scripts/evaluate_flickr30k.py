"""评测 CLIP 与 LongCLIP-B 在 Flickr30k 上的零样本图文检索结果。"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval import (  # noqa: E402
    RetrievalItem,
    compute_retrieval_metrics,
    flatten_captions,
    normalize,
    read_items,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        命令行参数对象。
    """

    parser = argparse.ArgumentParser(description="Flickr30k 零样本图文检索评测。")
    parser.add_argument("--annotations-jsonl", type=Path, required=True, help="Flickr30k JSONL 标注。")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--longclip-checkpoint", type=Path, required=True)
    parser.add_argument("--longclip-root", type=Path, default=REPO_ROOT / "third_party" / "Long-CLIP")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-images", type=int, default=0, help="0 表示使用全部图片。")
    parser.add_argument("--batch-size-images", type=int, default=128)
    parser.add_argument("--batch-size-texts", type=int, default=256)
    parser.add_argument("--skip-clip", action="store_true", help="跳过 Transformers CLIP baseline。")
    parser.add_argument("--skip-longclip", action="store_true", help="跳过 LongCLIP-B。")
    return parser.parse_args()


def select_device(device_name: str) -> torch.device:
    """选择运行设备。

    Args:
        device_name: auto、cpu、cuda 或 cuda:0。

    Returns:
        PyTorch 设备。
    """

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def batched(items: list[str | Path], batch_size: int) -> list[list[str | Path]]:
    """将列表切分为 batch。

    Args:
        items: 输入列表。
        batch_size: batch 大小。

    Returns:
        切分后的列表。
    """

    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def encode_clip_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """使用 Transformers CLIP 编码图像。

    Args:
        model: CLIP 模型。
        processor: CLIP 预处理器。
        image_paths: 图像路径。
        batch_size: batch 大小。
        device: 运行设备。

    Returns:
        图像特征矩阵。
    """

    features: list[torch.Tensor] = []
    for batch_paths in tqdm(batched(image_paths, batch_size), desc="clip encode images"):
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        try:
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                batch_features = model.get_image_features(**inputs)
            features.append(normalize(batch_features).cpu())
        finally:
            for image in images:
                image.close()
    return torch.cat(features, dim=0)


def encode_clip_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """使用 Transformers CLIP 编码文本。

    Args:
        model: CLIP 模型。
        processor: CLIP 预处理器。
        texts: 文本列表。
        batch_size: batch 大小。
        device: 运行设备。

    Returns:
        文本特征矩阵。
    """

    features: list[torch.Tensor] = []
    for batch_texts in tqdm(batched(texts, batch_size), desc="clip encode texts"):
        inputs = processor(
            text=batch_texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            batch_features = model.get_text_features(**inputs)
        features.append(normalize(batch_features).cpu())
    return torch.cat(features, dim=0)


def load_longclip_module(longclip_root: Path) -> object:
    """加载官方 Long-CLIP 模块。

    Args:
        longclip_root: 官方 Long-CLIP 仓库路径。

    Returns:
        `model.longclip` 模块对象。

    Raises:
        FileNotFoundError: 官方仓库路径不存在。
    """

    if not longclip_root.exists():
        raise FileNotFoundError(f"Long-CLIP 仓库不存在: {longclip_root}")
    sys.path.insert(0, str(longclip_root))
    return importlib.import_module("model.longclip")


def encode_longclip_images(
    model: torch.nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    image_paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """使用 LongCLIP-B 编码图像。

    Args:
        model: LongCLIP-B 模型。
        preprocess: 官方图像预处理函数。
        image_paths: 图像路径。
        batch_size: batch 大小。
        device: 运行设备。

    Returns:
        图像特征矩阵。
    """

    features: list[torch.Tensor] = []
    for batch_paths in tqdm(batched(image_paths, batch_size), desc="longclip encode images"):
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        try:
            image_input = torch.stack([preprocess(image) for image in images]).to(device)
            with torch.inference_mode():
                batch_features = model.encode_image(image_input)
            features.append(normalize(batch_features).cpu())
        finally:
            for image in images:
                image.close()
    return torch.cat(features, dim=0)


def encode_longclip_texts(
    model: torch.nn.Module,
    longclip_module: object,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """使用 LongCLIP-B 编码文本。

    Args:
        model: LongCLIP-B 模型。
        longclip_module: 官方 `model.longclip` 模块。
        texts: 文本列表。
        batch_size: batch 大小。
        device: 运行设备。

    Returns:
        文本特征矩阵。
    """

    features: list[torch.Tensor] = []
    for batch_texts in tqdm(batched(texts, batch_size), desc="longclip encode texts"):
        tokens = longclip_module.tokenize(batch_texts, truncate=True).to(device)
        with torch.inference_mode():
            batch_features = model.encode_text(tokens)
        features.append(normalize(batch_features).cpu())
    return torch.cat(features, dim=0)


def build_output(
    model_name: str,
    annotations_jsonl: Path,
    items: list[RetrievalItem],
    metrics: dict[str, float],
) -> dict[str, object]:
    """构建 JSON 输出对象。

    Args:
        model_name: 模型名称。
        annotations_jsonl: 标注文件路径。
        items: 检索样本。
        metrics: 指标字典。

    Returns:
        可序列化输出对象。
    """

    num_captions = sum(len(item.captions) for item in items)
    return {
        "model_name": model_name,
        "annotations_jsonl": str(annotations_jsonl),
        "num_images": len(items),
        "num_captions": num_captions,
        "metrics": metrics,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    """写入 JSON 文件。

    Args:
        path: 输出路径。
        payload: 输出内容。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_clip(
    args: argparse.Namespace,
    items: list[RetrievalItem],
    texts: list[str],
    caption_counts: list[int],
    device: torch.device,
) -> dict[str, object]:
    """评测 Transformers CLIP baseline。

    Args:
        args: 命令行参数。
        items: 检索样本。
        texts: 文本列表。
        caption_counts: 每张图像的文本数量。
        device: 运行设备。

    Returns:
        模型评测输出。
    """

    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model.eval()
    image_features = encode_clip_images(
        model=model,
        processor=processor,
        image_paths=[item.image_path for item in items],
        batch_size=args.batch_size_images,
        device=device,
    )
    text_features = encode_clip_texts(
        model=model,
        processor=processor,
        texts=texts,
        batch_size=args.batch_size_texts,
        device=device,
    )
    metrics = compute_retrieval_metrics(image_features, text_features, caption_counts)
    del model
    torch.cuda.empty_cache()
    return build_output(args.clip_model, args.annotations_jsonl, items, metrics)


def evaluate_longclip(
    args: argparse.Namespace,
    items: list[RetrievalItem],
    texts: list[str],
    caption_counts: list[int],
    device: torch.device,
) -> dict[str, object]:
    """评测 LongCLIP-B。

    Args:
        args: 命令行参数。
        items: 检索样本。
        texts: 文本列表。
        caption_counts: 每张图像的文本数量。
        device: 运行设备。

    Returns:
        模型评测输出。
    """

    longclip_module = load_longclip_module(args.longclip_root)
    model, preprocess = longclip_module.load(str(args.longclip_checkpoint), device=device)
    model.eval()
    image_features = encode_longclip_images(
        model=model,
        preprocess=preprocess,
        image_paths=[item.image_path for item in items],
        batch_size=args.batch_size_images,
        device=device,
    )
    text_features = encode_longclip_texts(
        model=model,
        longclip_module=longclip_module,
        texts=texts,
        batch_size=args.batch_size_texts,
        device=device,
    )
    metrics = compute_retrieval_metrics(image_features, text_features, caption_counts)
    del model
    torch.cuda.empty_cache()
    return build_output("LongCLIP-B", args.annotations_jsonl, items, metrics)


def main() -> None:
    """运行评测。"""

    args = parse_args()
    device = select_device(args.device)
    items = read_items(args.annotations_jsonl, args.max_images)
    texts, caption_counts = flatten_captions(items)
    outputs: dict[str, dict[str, object]] = {}

    if not args.skip_clip:
        clip_output = evaluate_clip(args, items, texts, caption_counts, device)
        write_json(args.output_dir / "clip_flickr30k_metrics.json", clip_output)
        outputs["clip"] = clip_output
        print(json.dumps(clip_output, ensure_ascii=False, indent=2))

    if not args.skip_longclip:
        longclip_output = evaluate_longclip(args, items, texts, caption_counts, device)
        write_json(args.output_dir / "longclip_b_flickr30k_metrics.json", longclip_output)
        outputs["longclip_b"] = longclip_output
        print(json.dumps(longclip_output, ensure_ascii=False, indent=2))

    summary = {
        "annotations_jsonl": str(args.annotations_jsonl),
        "num_images": len(items),
        "num_captions": len(texts),
        "outputs": outputs,
    }
    write_json(args.output_dir / "summary_flickr30k_metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
