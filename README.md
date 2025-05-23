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
