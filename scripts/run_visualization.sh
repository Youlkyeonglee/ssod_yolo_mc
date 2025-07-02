#!/bin/bash

# MC Dropout Pseudo Label Distribution Visualization Script
# SSOD YOLO MC 프로젝트

echo "🎨 MC Dropout Pseudo Label 분포 시각화 시작"
echo "================================================"

# 기본 설정
CONFIG_FILE="src/configs/yolo_config.yaml"
SAVE_DIR="./visualization_results"
DEVICE="cuda:0"
MAX_BATCHES=20

# 명령행 인자 처리
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --max_batches)
            MAX_BATCHES="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --uncertainty)
            INCLUDE_UNCERTAINTY="--include_uncertainty_plots"
            shift
            ;;
        --help)
            echo "사용법: $0 [옵션]"
            echo "옵션:"
            echo "  --config <path>       YAML 설정 파일 경로 (기본: $CONFIG_FILE)"
            echo "  --save_dir <path>     결과 저장 디렉토리 (기본: $SAVE_DIR)"
            echo "  --device <device>     실행 디바이스 (기본: $DEVICE)"
            echo "  --max_batches <num>   최대 배치 수 (기본: $MAX_BATCHES)"
            echo "  --checkpoint <path>   사전 학습된 모델 체크포인트"
            echo "  --uncertainty         불확실성 분포 그래프 포함"
            echo "  --help                이 도움말 표시"
            exit 0
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            exit 1
            ;;
    esac
done

# 디렉토리 확인
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 설정 파일을 찾을 수 없습니다: $CONFIG_FILE"
    exit 1
fi

# 결과 디렉토리 생성
mkdir -p "$SAVE_DIR"

# GPU 확인
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 GPU 상태 확인:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits
    echo ""
fi

# 시각화 실행
echo "📊 시각화 파라미터:"
echo "  - 설정 파일: $CONFIG_FILE"
echo "  - 저장 디렉토리: $SAVE_DIR"
echo "  - 디바이스: $DEVICE"
echo "  - 최대 배치: $MAX_BATCHES"
if [ ! -z "$CHECKPOINT" ]; then
    echo "  - 체크포인트: $CHECKPOINT"
fi
if [ ! -z "$INCLUDE_UNCERTAINTY" ]; then
    echo "  - 불확실성 그래프: 포함"
fi
echo ""

# 메인 실행 명령
CMD="cd src && uv run distribution_visualization.py --config $CONFIG_FILE --save_dir $SAVE_DIR --device $DEVICE --max_batches $MAX_BATCHES"

if [ ! -z "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

if [ ! -z "$INCLUDE_UNCERTAINTY" ]; then
    CMD="$CMD $INCLUDE_UNCERTAINTY"
fi

echo "🚀 실행 명령: $CMD"
echo ""

# 실행
eval $CMD

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 시각화 완료!"
    echo "📁 결과 파일:"
    ls -la "$SAVE_DIR"
    echo ""
    echo "📊 생성된 그래프:"
    echo "  - class_distribution.png: 클래스별 분포"
    echo "  - confidence_iou_scatter.png: Confidence-IoU 분산도"
    if [ ! -z "$INCLUDE_UNCERTAINTY" ]; then
        echo "  - uncertainty_distributions.png: 불확실성 분포"
        echo "  - confidence_calibration.png: 캘리브레이션 곡선"
    fi
    echo "  - summary_statistics.txt: 상세 통계 보고서"
else
    echo "❌ 시각화 실행 실패"
    exit 1
fi 