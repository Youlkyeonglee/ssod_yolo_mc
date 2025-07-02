# SSOD YOLO MC - 4장 GPU 분산 학습 가이드

본 가이드는 4장의 GPU를 활용한 분산 병렬 학습 방법을 설명합니다.

## 🚀 빠른 시작

### 1. 자동 GPU 감지로 4장 GPU 사용
```bash
# src 디렉토리에서 실행
cd src
uv run train.py --use_distributed
```

### 2. 특정 GPU 지정하여 사용
```bash
cd src
uv run train.py --gpu_ids 0,1,2,3 --use_distributed
```

### 3. 분산 학습 전용 스크립트 사용
```bash
cd src
uv run train_distributed.py --distributed --gpu_ids 0,1,2,3
```

## ⚙️ 설정 방법

### yolo_config.yaml에서 GPU 설정
```yaml
gpu:
  # GPU 사용 설정
  use_gpu: true                    # GPU 사용 여부
  auto_detect: true                # 자동 GPU 감지 (true일 경우 gpu_ids 무시)
  gpu_ids: [0, 1, 2, 3]           # 사용할 GPU ID 리스트
  
  # 분산 학습 설정
  distributed:
    enabled: true                  # 분산 학습 활성화
    backend: "nccl"               # 분산 백엔드
    master_addr: "localhost"      # 마스터 노드 주소
    master_port: "12355"          # 마스터 노드 포트
    
  # 메모리 관리
  memory:
    mixed_precision: true         # 혼합 정밀도 학습 (AMP)
    gradient_accumulation: 1      # 그래디언트 누적 단계
    pin_memory: true              # DataLoader pin_memory 설정
    non_blocking: true            # 비동기 GPU 전송
```

## 📋 명령행 옵션

### train.py 옵션
- `--use_distributed`: 분산 학습 강제 활성화
- `--gpu_ids 0,1,2,3`: 사용할 GPU ID 지정
- `--config path/to/config.yaml`: 설정 파일 경로
- `--save_dir path/to/save`: 결과 저장 디렉토리

### train_distributed.py 추가 옵션
- `--distributed`: 분산 학습 활성화
- `--world_size 4`: 전체 프로세스 수
- `--master_addr localhost`: 마스터 노드 주소
- `--master_port 12355`: 마스터 노드 포트

## 🔧 사용 예시

### 1. 기본 4장 GPU 분산 학습
```bash
cd src
uv run train.py \
    --use_distributed \
    --config configs/yolo_config.yaml \
    --save_dir ../runs
```

### 2. 특정 2장 GPU만 사용
```bash
cd src
uv run train.py \
    --gpu_ids 0,1 \
    --use_distributed \
    --config configs/yolo_config.yaml
```

### 3. 단일 GPU 사용 (GPU 0번만)
```bash
cd src
uv run train.py \
    --gpu_ids 0 \
    --config configs/yolo_config.yaml
```

### 4. CPU 사용
```bash
cd src
uv run train.py \
    --device cpu \
    --config configs/yolo_config.yaml
```

## 📊 GPU 사용량 모니터링

### 실시간 GPU 상태 확인
```bash
# 별도 터미널에서 실행
watch -n 1 nvidia-smi
```

### GPU 메모리 사용량 확인
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv --loop=1
```

## 🔍 성능 최적화 팁

### 1. 배치 크기 조정
```yaml
data:
  batch_size: 8  # GPU당 배치 크기 (4장 GPU면 총 32)
```

### 2. 워커 프로세스 수 조정
```yaml
data:
  num_workers: 4  # GPU당 워커 수
```

### 3. 혼합 정밀도 학습 활성화
```yaml
gpu:
  memory:
    mixed_precision: true  # 메모리 절약 + 속도 향상
```

### 4. 그래디언트 누적 사용 (메모리 부족 시)
```yaml
gpu:
  memory:
    gradient_accumulation: 2  # 2번 누적 후 업데이트
```

## 🚨 문제 해결

### GPU 메모리 부족 오류
1. 배치 크기 감소: `batch_size: 4`
2. 워커 수 감소: `num_workers: 2`
3. 그래디언트 누적 활성화
4. 혼합 정밀도 학습 활성화

### 분산 학습 연결 오류
1. 포트 변경: `master_port: "12356"`
2. 방화벽 확인
3. GPU 드라이버 업데이트

### 성능이 느린 경우
1. NCCL 백엔드 사용 확인: `backend: "nccl"`
2. pin_memory 활성화: `pin_memory: true`
3. non_blocking 활성화: `non_blocking: true`

## 📈 예상 성능 향상

| GPU 수 | 상대적 학습 속도 | 메모리 효율성 |
|--------|------------------|---------------|
| 1장    | 1.0x            | 기준         |
| 2장    | 1.8x            | 2배          |
| 4장    | 3.5x            | 4배          |

## 🔧 고급 설정

### 다중 노드 분산 학습 (미래 확장)
```bash
# 마스터 노드
uv run train_distributed.py --distributed --master_addr 192.168.1.100

# 워커 노드
uv run train_distributed.py --distributed --master_addr 192.168.1.100 --rank 1
```

### 환경 변수 설정
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # InfiniBand 비활성화 (필요시)
```

## 📝 로그 확인

분산 학습 시 로그는 메인 프로세스(rank 0)에서만 출력됩니다.

### 학습 진행 상황 확인
```bash
tail -f runs/train/yolo_*/training.log
```

### GPU 사용량 확인
```bash
nvidia-smi dmon -s pucvmet
```

---

## 💡 추천 워크플로우

1. **단일 GPU로 테스트**: 설정 검증
2. **2장 GPU로 확인**: 분산 학습 동작 확인  
3. **4장 GPU로 본격 학습**: 최대 성능 활용
4. **결과 모니터링**: GPU 사용률 및 학습 진행 상황 확인

## 📞 지원

문제 발생 시 다음 정보와 함께 문의:
- GPU 모델 및 드라이버 버전
- PyTorch 버전
- 오류 로그
- 사용한 명령어 