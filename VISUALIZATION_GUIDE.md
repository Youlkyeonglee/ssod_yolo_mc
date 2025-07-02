# MC Dropout Pseudo Label 분포 시각화 가이드

MC Dropout의 효과를 확인하기 위해 unlabeled data의 pseudo label 분포를 시각화하는 도구입니다.

## 🎯 주요 기능

### 생성되는 시각화

1. **클래스 분포 그래프** (`class_distribution.png`)
   - x축: COCO 클래스명
   - y축: 검출 개수
   - 데이터: Labeled Pseudo, Unlabeled Pseudo, Val Pseudo, Ground Truth

2. **Confidence-IoU 분산도** (`confidence_iou_scatter.png`)
   - x축: Classification Confidence
   - y축: IoU with Ground Truth
   - 신뢰도와 실제 정확도 간의 상관관계 분석

3. **불확실성 분포 히스토그램** (`uncertainty_distributions.png`, 옵션)
   - Box Variance 분포
   - Class Entropy 분포
   - Box Variance vs Class Entropy 산점도
   - 데이터 소스별 Box Variance 박스플롯

4. **신뢰도 캘리브레이션 곡선** (`confidence_calibration.png`, 옵션)
   - Reliability Diagram (캘리브레이션 곡선)
   - Confidence 분포 히스토그램
   - Expected Calibration Error (ECE) 계산

5. **상세 통계 보고서** (`summary_statistics.txt`)
   - 전체 데이터 통계
   - 클래스별 검출 통계
   - Confidence 통계
   - 불확실성 통계 (MC Dropout)
   - 분석 권장사항

## 🚀 사용 방법

### 1. 기본 실행

```bash
# 스크립트 사용 (권장)
./scripts/run_visualization.sh

# 직접 실행
cd src && uv run distribution_visualization.py
```

### 2. 고급 옵션

```bash
# 특정 설정 파일 사용
./scripts/run_visualization.sh --config src/configs/yolo_config_visualization.yaml

# 불확실성 분포 그래프 포함
./scripts/run_visualization.sh --uncertainty

# 사전 학습된 모델 체크포인트 사용
./scripts/run_visualization.sh --checkpoint runs/train/exp1/best.pt

# 다른 GPU 사용
./scripts/run_visualization.sh --device cuda:1

# 분석할 배치 수 조정
./scripts/run_visualization.sh --max_batches 50

# 결과 저장 위치 변경
./scripts/run_visualization.sh --save_dir ./my_results
```

### 3. 전체 옵션 조합

```bash
./scripts/run_visualization.sh \
    --config src/configs/yolo_config.yaml \
    --checkpoint runs/train/best_model.pt \
    --save_dir ./detailed_analysis \
    --device cuda:0 \
    --max_batches 30 \
    --uncertainty
```

## ⚙️ 설정 파일 수정

`src/configs/yolo_config_visualization.yaml` 파일에서 다음 항목들을 조정할 수 있습니다:

### 모델 설정
```yaml
model:
  name: "yolov8n"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  dropout:
    rate: 0.1      # 0.05 ~ 0.3
    num_samples: 5 # MC 샘플링 횟수 (5, 10, 20)
```

### 데이터 설정
```yaml
data:
  batch_size: 4      # 메모리에 맞게 조정
  max_samples: 100   # 빠른 테스트시 작게
  labeled_ratio: 1.0 # 1.0, 2.0, 5.0, 10.0
  seed: 1           # 재현성을 위한 시드
```

### 불확실성 임계값
```yaml
training:
  semi_supervised:
    uncertainty:
      box_std_threshold: 0.1   # 낮을수록 엄격한 필터링
      entropy_threshold: 0.5   # 낮을수록 엄격한 필터링
```

### COCO 데이터셋 경로
```yaml
data:
  coco:
    train_root: "/path/to/COCO/train2017"
    val_root: "/path/to/COCO/val2017"
    train_anno: "/path/to/COCO/annotations/instances_train2017.json"
    val_anno: "/path/to/COCO/annotations/instances_val2017.json"
```

## 📊 결과 해석 가이드

### 1. 클래스 분포 분석
- **불균형 정도**: 클래스 간 검출 개수 차이 확인
- **GT vs Pseudo**: Ground Truth와 Pseudo Label 분포 비교
- **데이터 소스별 차이**: Labeled, Unlabeled, Val 데이터의 분포 차이

### 2. Confidence-IoU 분석
- **이상적인 경우**: 대각선에 가까운 분포 (confidence = accuracy)
- **과신**: 대각선 위쪽 (높은 confidence, 낮은 IoU)
- **과소평가**: 대각선 아래쪽 (낮은 confidence, 높은 IoU)

### 3. 불확실성 분석
- **Box Variance**: 높을수록 위치 예측의 불확실성이 큼
- **Class Entropy**: 높을수록 클래스 분류의 불확실성이 큼
- **필터링 전략**: 높은 불확실성 샘플 제거 고려

### 4. 캘리브레이션 품질
- **ECE (Expected Calibration Error)**: 낮을수록 좋음 (< 0.1 권장)
- **Perfect Calibration**: 캘리브레이션 곡선이 대각선에 가까움
- **Under/Over-confident**: 곡선이 대각선에서 벗어남

## 🔧 문제 해결

### 메모리 부족 오류
```bash
# 배치 크기 줄이기
./scripts/run_visualization.sh --max_batches 10

# 더 작은 모델 사용 (config 파일에서 yolov8n 으로 변경)
```

### GPU 메모리 부족
```bash
# CPU 사용
./scripts/run_visualization.sh --device cpu

# 작은 이미지 크기 사용 (config 파일에서 img_size: 416)
```

### 데이터 로딩 오류
```bash
# COCO 데이터셋 경로 확인
# config 파일의 data.coco 섹션 수정
```

### 체크포인트 로딩 오류
```bash
# 체크포인트 파일 경로 확인
ls runs/train/*/weights/best.pt

# 올바른 경로로 지정
./scripts/run_visualization.sh --checkpoint runs/train/exp1/weights/best.pt
```

## 📋 요구사항

### 필수 의존성
- Python 3.11+
- PyTorch
- Ultralytics YOLO
- matplotlib, seaborn
- numpy, pandas
- tqdm

### 하드웨어 권장사항
- GPU: NVIDIA GTX 1060 이상 (4GB+ VRAM)
- RAM: 8GB 이상
- 저장공간: 5GB 이상 (COCO 데이터셋 포함)

## 🎯 사용 사례

### 1. 빠른 프로토타이핑
```bash
# 경량 설정으로 빠른 테스트
./scripts/run_visualization.sh \
    --config src/configs/yolo_config_visualization.yaml \
    --max_batches 5
```

### 2. 상세 분석
```bash
# 모든 그래프 포함한 완전한 분석
./scripts/run_visualization.sh \
    --uncertainty \
    --max_batches 50 \
    --checkpoint best_model.pt
```

### 3. 다양한 설정 비교
```bash
# 여러 설정으로 비교 실험
for config in baseline proposed; do
    ./scripts/run_visualization.sh \
        --config src/configs/yolo_config_${config}.yaml \
        --save_dir ./results_${config} \
        --uncertainty
done
```

## 📈 성능 최적화 팁

1. **MC 샘플 수 조정**: 정확도 vs 속도 trade-off
2. **배치 크기 최적화**: GPU 메모리에 맞게 조정
3. **이미지 크기 줄이기**: 빠른 프로토타이핑시 416x416 사용
4. **데이터 샘플 제한**: max_samples로 분석 데이터 수 조정
5. **병렬 처리**: num_workers 증가 (CPU 코어 수에 맞게)

---

더 자세한 정보는 프로젝트의 메인 README.md와 기술 문서를 참고하세요. 