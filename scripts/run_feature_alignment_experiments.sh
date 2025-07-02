#!/bin/bash

# 🎯 Feature Alignment 효과 검증 실험 스크립트
# Baseline (Feature Alignment 없음) vs Proposed (Feature Alignment 포함) 비교

set -e

echo "🚀 Feature Alignment 효과 검증 실험 시작"
echo "========================================"

# 실험 설정
PROJECT_ROOT="/media/oem/personal_vol/yklee/ssod_yolo_mc"
SRC_DIR="${PROJECT_ROOT}/src"
RESULTS_DIR="${PROJECT_ROOT}/results/feature_alignment_experiments"

# 결과 디렉토리 생성
mkdir -p "$RESULTS_DIR"

cd "$SRC_DIR"

echo "📍 현재 작업 디렉토리: $(pwd)"
echo "📁 결과 저장 디렉토리: $RESULTS_DIR"

# 1. Baseline 실험 (Feature Alignment 비활성화)
echo ""
echo "🔥 실험 1: Baseline (Feature Alignment 비활성화)"
echo "================================================"
echo "⚙️  설정: configs/yolo_config_baseline.yaml"
echo "📝 설명: Feature Alignment Loss 없이 기본 Teacher-Student 모델"

uv run train.py \
    --config configs/yolo_config_baseline.yaml \
    --save_dir "$RESULTS_DIR/baseline" \
    --use_distributed \
    2>&1 | tee "$RESULTS_DIR/baseline_experiment.log"

echo "✅ Baseline 실험 완료"

# 결과 백업
mv runs/train "$RESULTS_DIR/baseline_runs" || echo "⚠️  runs/train 디렉토리가 없습니다"

# 2. Proposed 실험 (Feature Alignment 활성화)
echo ""
echo "🔥 실험 2: Proposed (Feature Alignment 활성화)"
echo "==============================================="
echo "⚙️  설정: configs/yolo_config_feature_alignment.yaml"
echo "📝 설명: Feature Alignment Loss 포함한 완전한 Teacher-Student 모델"

uv run train.py \
    --config configs/yolo_config_feature_alignment.yaml \
    --save_dir "$RESULTS_DIR/proposed" \
    --use_distributed \
    2>&1 | tee "$RESULTS_DIR/proposed_experiment.log"

echo "✅ Proposed 실험 완료"

# 결과 백업
mv runs/train "$RESULTS_DIR/proposed_runs" || echo "⚠️  runs/train 디렉토리가 없습니다"

# 3. 결과 비교 분석
echo ""
echo "📊 실험 결과 분석"
echo "=================="

# 실험 결과 요약 생성
cat > "$RESULTS_DIR/experiment_summary.md" << EOF
# Feature Alignment 효과 검증 실험 결과

## 실험 개요
- **목적**: Feature Alignment Loss의 효과 검증
- **데이터셋**: COCO 2017 (labeled_ratio: 10%, seed: 1)
- **모델**: YOLOv8m with MC Dropout Teacher-Student
- **실험 날짜**: $(date)

## 실험 설정

### Baseline (Feature Alignment 비활성화)
- **설정 파일**: configs/yolo_config_baseline.yaml
- **Feature Alignment**: enabled: false, weight: 0.0
- **로그 파일**: baseline_experiment.log
- **결과 디렉토리**: baseline_runs/

### Proposed (Feature Alignment 활성화)
- **설정 파일**: configs/yolo_config_feature_alignment.yaml
- **Feature Alignment**: enabled: true, weight: 0.1
- **로그 파일**: proposed_experiment.log
- **결과 디렉토리**: proposed_runs/

## 평가 지표
- **mAP@0.5**: IoU 0.5에서의 평균 정밀도
- **mAP@0.5:0.95**: IoU 0.5-0.95 범위 평균 정밀도
- **Feature Alignment Loss**: Teacher-Student feature 정합 손실
- **Training Loss**: 전체 학습 손실 추이
- **Pseudo Label Quality**: MC Dropout 기반 의사 레이블 품질

## 결과 분석
### Baseline vs Proposed 성능 비교
- [ ] mAP 성능 차이 분석
- [ ] Feature alignment loss 추이 확인
- [ ] 학습 안정성 평가
- [ ] Pseudo label 품질 개선 효과

### 실험 결론
- [ ] Feature Alignment의 효과 유무
- [ ] 성능 향상 정도 정량화
- [ ] 향후 연구 방향 제시

EOF

echo "📋 실험 요약 파일 생성: $RESULTS_DIR/experiment_summary.md"

# 실험 완료 알림
echo ""
echo "🎉 Feature Alignment 효과 검증 실험 완료!"
echo "========================================"
echo "📁 모든 결과는 다음 위치에 저장되었습니다:"
echo "   $RESULTS_DIR"
echo ""
echo "📊 다음 파일들을 확인하세요:"
echo "   - baseline_experiment.log: Baseline 실험 로그"
echo "   - proposed_experiment.log: Proposed 실험 로그"
echo "   - experiment_summary.md: 실험 결과 요약"
echo "   - baseline_runs/: Baseline 모델 체크포인트 및 메트릭"
echo "   - proposed_runs/: Proposed 모델 체크포인트 및 메트릭"
echo ""
echo "🔍 성능 비교를 위해 각 runs 디렉토리의 메트릭 파일을 확인하세요!" 