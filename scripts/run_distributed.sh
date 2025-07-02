#!/bin/bash

# SSOD YOLO MC 분산 학습 실행 스크립트
# 4개 GPU (0,1,2,3)를 사용한 분산 학습

echo "=== SSOD YOLO MC 분산 학습 시작 ==="

# GPU 정보 확인
echo "GPU 정보 확인:"
nvidia-smi

# 현재 디렉토리 확인
echo "현재 디렉토리: $(pwd)"

# Python 환경 확인
echo "Python 버전: $(python --version)"
echo "PyTorch 버전: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA 사용 가능: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU 개수: $(python -c 'import torch; print(torch.cuda.device_count())')"

# 설정 파일 경로
CONFIG_PATH="src/configs/yolo_config.yaml"

# 기본 실행 (4개 GPU 자동 감지)
echo ""
echo "=== 4개 GPU 자동 감지 분산 학습 ==="
cd src

# UV를 사용한 실행
echo "UV 환경에서 분산 학습 시작..."
uv run train.py \
    --use_distributed \
    --config configs/yolo_config.yaml \
    --save_dir ../runs

echo "분산 학습 완료!"

# 대안: train_distributed.py 사용
echo ""
echo "=== 대안: 분산 학습 전용 스크립트 ==="
uv run train_distributed.py \
    --distributed \
    --gpu_ids 0,1,2,3 \
    --config configs/yolo_config.yaml \
    --save_dir ../runs

echo "모든 분산 학습 완료!" 