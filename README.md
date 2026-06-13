# Long-CLIP 图文检索复现

本项目复现 Long-CLIP 在图文检索任务上的主实验流程，比较 `CLIP ViT-B/32` 与 `LongCLIP-B` 在 Flickr30k test split 上的零样本检索结果。

## 模型与数据

| 项目 | 内容 |
| --- | --- |
| 论文 | Long-CLIP: Unlocking the Long-Text Capability of CLIP |
| 官方代码 | https://github.com/beichenzbc/Long-CLIP |
| LongCLIP-B checkpoint | https://huggingface.co/BeichenZhang/LongCLIP-B |
| 数据集 | Flickr30k test split |
| 标注文件 | `/mnt/kxh/smx/homework/h3/datasets/flickr30k/annotations.jsonl` |

## 环境

```bash
/mnt/kxh/miniconda3/envs/trl/bin/python
torch 2.8.0+cu128
torchvision 0.23.0+cu128
transformers 4.57.6
```

Long-CLIP tokenizer 需要 `ftfy`：

```bash
/mnt/kxh/miniconda3/envs/trl/bin/python -m pip install ftfy
```

## 准备代码与权重

```bash
cd /mnt/kxh/smx/homework/h4
mkdir -p checkpoints third_party
git clone https://github.com/beichenzbc/Long-CLIP.git third_party/Long-CLIP
git -C third_party/Long-CLIP checkout 3966af9ae9331666309a22128468b734db4672a7
aria2c -c -x 8 -s 8 -k 1M -d checkpoints -o longclip-B.pt \
  'https://hf-mirror.com/BeichenZhang/LongCLIP-B/resolve/main/longclip-B.pt'
```

## 运行评测

Smoke test：

```bash
cd /mnt/kxh/smx/homework/h4
CUDA_VISIBLE_DEVICES=6 bash scripts/run_smoke.sh
```

完整 Flickr30k test 评测：

```bash
cd /mnt/kxh/smx/homework/h4
CUDA_VISIBLE_DEVICES=6 bash scripts/run_full.sh
```

直接调用评测脚本：

```bash
cd /mnt/kxh/smx/homework/h4
CUDA_VISIBLE_DEVICES=6 /mnt/kxh/miniconda3/envs/trl/bin/python scripts/evaluate_flickr30k.py \
  --annotations-jsonl /mnt/kxh/smx/homework/h3/datasets/flickr30k/annotations.jsonl \
  --clip-model openai/clip-vit-base-patch32 \
  --longclip-checkpoint checkpoints/longclip-B.pt \
  --output-dir results
```

## 输出

评测脚本会输出：

- `results/clip_flickr30k_metrics.json`
- `results/longclip_b_flickr30k_metrics.json`
- `results/summary_flickr30k_metrics.json`

指标包括：

- `image_to_text_r1`
- `image_to_text_r5`
- `image_to_text_r10`
- `text_to_image_r1`
- `text_to_image_r5`
- `text_to_image_r10`
- `rsum`
