# Semi-Supervised Object Detection with MC Dropout

본 프로젝트는 MC Dropout을 활용한 Semi-Supervised Object Detection 구현입니다.

## 프로젝트 구조

```
src/
├── models/
│   └── yolo_mc.py         # MC Dropout이 적용된 YOLO 모델
├── data/
│   └── semi_supervised_dataset.py  # 데이터 로더
├── utils/
│   └── train_utils.py     # 학습 유틸리티 함수
├── uncertainty/
│   └── mc_dropout.py      # 불확실성 추정
└── train.py               # 학습 스크립트
```

## 학습 프로세스

1. **데이터 준비**
   - 전체 데이터셋을 레이블된 데이터와 레이블되지 않은 데이터로 분할
   - 레이블된 데이터 비율: 1%, 2%, 5%, 10% 지원
   - YOLO 형식의 레이블 사용: `class_id x_center y_center width height`

2. **모델 초기화**
   - YOLOv8 모델 로드 및 MC Dropout 레이어 추가
   - Dropout rate 설정 (기본값: 0.1)
   - 모델을 학습 모드로 설정하고 gradient 계산 활성화

3. **데이터 로더 설정**
   - 레이블된 데이터 로더
   - 레이블되지 않은 데이터 로더
   - 검증 데이터 로더
   - Albumentations를 사용한 데이터 증강 적용

4. **학습 루프**
   - 레이블된 데이터를 사용한 지도 학습
   - MC Dropout을 통한 불확실성 추정
   - 불확실성이 낮은 샘플에 대한 의사 레이블 생성
   - 의사 레이블을 사용한 반지도 학습
   - 손실 함수 계산 및 역전파

5. **모델 평가**
   - 주기적인 검증 데이터셋 평가
   - mAP50 및 mAP50-95 메트릭 계산
   - 최고 성능 모델 저장

6. **결과 저장**
   - 실행 결과는 `runs` 디렉토리에 저장
   - 폴더명 형식: `{model_name}_label_p{labeled_ratio}`
   - 중복 실행 시 자동 넘버링 (예: yolov8m_label_p10.0_1)
   - 체크포인트 및 로그 파일 저장

## 실행 방법

```bash
# 일반 실행
uv run src/train.py \
    --labeled-ratio 10.0 \
    --batch-size 16 \
    --epochs 12 \
    --model yolov8m \
    --dropout-rate 0.1

# 테스트 실행 (100개 샘플만 사용)
uv run src/train.py \
    --labeled-ratio 10.0 \
    --batch-size 16 \
    --epochs 12 \
    --model yolov8m \
    --dropout-rate 0.1 \
    --max-samples 100
```

## 주요 매개변수

- `labeled-ratio`: 레이블된 데이터 비율 (1.0, 2.0, 5.0, 10.0)
- `batch-size`: 배치 크기
- `epochs`: 학습 에포크 수
- `model`: YOLO 모델 이름 (yolov8n, yolov8s, yolov8m 등)
- `dropout-rate`: MC Dropout 비율
- `num-samples`: MC Dropout 샘플링 횟수
- `box-std-threshold`: 박스 좌표 표준편차 임계값
- `entropy-threshold`: 클래스 엔트로피 임계값
- `val-interval`: 검증을 수행할 에포크 간격 (기본값: 1)
- `save-interval`: 체크포인트를 저장할 에포크 간격 (기본값: 10)
- `max-samples`: 학습에 사용할 최대 샘플 수 (테스트용, 기본값: None은 전체 데이터 사용)

## 결과 구조

```
runs/
├── yolov8m_label_p10.0/
│   ├── train.log          # 학습 로그
│   ├── best.pth           # 최고 성능 모델
│   └── checkpoint_epoch_*.pth  # 주기적 체크포인트
└── ...
```

## 의존성

- PyTorch
- Ultralytics (YOLOv8)
- Albumentations
- NumPy
- OpenCV
- PyYAML

# Semi-Supervised Object Detection with Uncertainty-aware Loss

본 프로젝트는 Semi-Supervised Object Detection을 위한 Uncertainty-aware Loss를 구현합니다.

## Loss Function

전체 손실 함수는 supervised loss와 unsupervised loss의 가중치 합으로 구성됩니다:

```
Total Loss = Supervised Loss + λ × Unsupervised Loss
```

### 1. Supervised Loss (레이블된 데이터)

기존 객체 검출기의 손실 함수를 그대로 사용합니다:

- Classification Loss: Focal Loss
  ```
  FL(p_t) = -α_t(1-p_t)^γ log(p_t)
  ```

- Localization Loss: CIoU Loss
  ```
  L_box = 1 - IoU + ρ²(b,b^gt)/c² + αv
  ```
  - ρ: 박스 중심점 간의 유클리드 거리
  - c: 두 박스를 포함하는 가장 작은 박스의 대각선 길이
  - v: 종횡비 일관성 측정
  - α: 양의 trade-off 파라미터

- Objectness Loss: Binary Cross Entropy Loss
  ```
  L_obj = -[y log(p) + (1-y)log(1-p)]
  ```

```
Supervised Loss = L_cls + L_box + L_obj
```

### 2. Unsupervised Loss (미레이블 데이터)

MC Dropout을 통해 추정된 예측 불확실성(variance)을 반영한 가중치(weight)를 적용합니다:

```
Unsupervised Loss = w × (L_cls + L_box + L_obj)
```

여기서 가중치 w는 다음과 같이 계산됩니다:

```
w = exp(-α × σ²)
```

- α: 가중치 감소 정도를 조절하는 하이퍼파라미터 (기본값: 50)
- σ²: MC Dropout 추론을 통해 얻어진 예측 결과의 분산(variance)

#### 불확실성 가중치의 특성

1. 예측이 확실할 때 (σ² → 0):
   - w → 1
   - 높은 신뢰도로 학습에 반영

2. 예측이 불확실할 때 (σ² → ∞):
   - w → 0
   - 낮은 신뢰도로 학습에 반영

## 구현 세부사항

### YOLO 손실 함수 구현

새로운 `yolo_losses.py` 모듈에서 다음 손실 함수들을 구현:

1. **Bounding Box Loss**:
   - **GIoU Loss**: Generalized Intersection over Union
   - **CIoU Loss**: Complete Intersection over Union (기본값)
   ```python
   # CIoU 계산 공식
   ciou = iou - (rho2 / c2 + v * alpha)
   loss = 1 - ciou
   ```

2. **Classification Loss**:
   - **Cross Entropy Loss**: 표준 분류 손실
   - **Focal Loss**: 클래스 불균형 해결 (γ > 0일 때)
   ```python
   focal_loss = (1 - pt)^γ * ce_loss
   ```

3. **Objectness Loss**:
   - **Binary Cross Entropy**: 객체 존재 여부 분류
   ```python
   obj_loss = F.binary_cross_entropy_with_logits(pred_obj, obj_targets)
   ```

### 손실 함수 설정

`yolo_config.yaml`에서 손실 함수 파라미터 설정:

```yaml
training:
  loss:
    supervised:
      box_loss_gain: 7.5      # Bbox loss weight
      cls_loss_gain: 0.5      # Classification loss weight  
      obj_loss_gain: 1.0      # Objectness loss weight
      bbox_loss_type: "ciou"  # "giou" or "ciou"
      focal_loss_gamma: 0.0   # 0.0 = standard CE, >0 = focal loss
      
    semi_supervised:
      box_loss_gain: 7.5
      cls_loss_gain: 0.5  
      obj_loss_gain: 1.0
      bbox_loss_type: "ciou"
      focal_loss_gamma: 0.0
      uncertainty_alpha: 50.0 # w = exp(-α × σ²)
```

### MC Dropout 및 불확실성 추정

1. MC Dropout을 통한 불확실성 추정:
   - Dropout rate: 0.1
   - MC sampling 횟수: 10
   - 예측 분산(σ²) 계산:
     ```python
     mc_predictions = [model(x) for _ in range(num_samples)]
     variance = torch.var(torch.stack(mc_predictions), dim=0)
     ```

2. 의사 레이블 생성 조건:
   - 신뢰도 임계값: 0.5
   - 박스 좌표 불확실성 임계값: 0.1
   - 클래스 엔트로피 임계값: 0.5

3. 학습 파라미터:
   - λ (pseudo_label_weight): 0.5
   - α (uncertainty_weight): 50.0
   - 의사 레이블링 시작 에포크: 10

### 사용 방법

```python
from src.utils.yolo_losses import create_yolo_loss

# Supervised learning용 손실 함수
loss_fn = create_yolo_loss(
    loss_type='supervised',
    bbox_loss_type='ciou',
    box_gain=7.5,
    cls_gain=0.5,
    obj_gain=1.0
)

# Semi-supervised learning용 손실 함수
semi_loss_fn = create_yolo_loss(
    loss_type='semi_supervised',
    bbox_loss_type='ciou', 
    uncertainty_alpha=50.0
)

# 손실 계산
box_loss, cls_loss, obj_loss = loss_fn(predictions, targets)
```


Phase 1 (현재): Conservative 설정으로 baseline 성능 확보
Phase 2: Focal Loss 도입 실험 (focal_loss_gamma: 1.5, 2.0)
Phase 3: Label Smoothing 실험 (label_smoothing: 0.1)
Phase 4: bbox_loss_gain 조정 실험 (5.0, 10.0, 15.0)