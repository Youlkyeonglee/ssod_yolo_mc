#!/bin/bash

# 🚫 Baseline 실험: Feature Alignment 비활성화
echo "🔥 Baseline 실험 (Feature Alignment 비활성화)"
echo "=============================================="

cd src

uv run train.py \
    --config configs/yolo_config_baseline.yaml \
    --use_distributed

echo "✅ Baseline 실험 완료" 