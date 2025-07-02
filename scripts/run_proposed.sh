#!/bin/bash

# ✅ Proposed 실험: Feature Alignment 활성화
echo "🔥 Proposed 실험 (Feature Alignment 활성화)"
echo "==========================================="

cd src

uv run train.py \
    --config configs/yolo_config_feature_alignment.yaml \
    --use_distributed

echo "✅ Proposed 실험 완료" 