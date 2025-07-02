#!/usr/bin/env python3

import torch
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent / "src"
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def test_gpu_availability():
    """GPU 사용 가능성 테스트"""
    print("=== GPU 테스트 ===")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}, 메모리: {props.total_memory // 1024**3}GB")
    else:
        print("GPU를 사용할 수 없습니다.")

def test_distributed_config():
    """분산 학습 설정 테스트"""
    print("\n=== 분산 학습 설정 테스트 ===")
    
    try:
        from utils.distributed_utils import setup_gpu_config, get_available_gpus
        import yaml
        
        # 테스트용 설정
        config = {
            'gpu': {
                'use_gpu': True,
                'auto_detect': True,
                'gpu_ids': [0, 1, 2, 3],
                'distributed': {
                    'enabled': True,
                    'backend': 'nccl',
                    'master_addr': 'localhost',
                    'master_port': '12355'
                },
                'memory': {
                    'mixed_precision': True,
                    'pin_memory': True
                }
            }
        }
        
        # GPU 설정 확인
        device, gpu_ids, use_distributed = setup_gpu_config(config)
        
        print(f"주요 디바이스: {device}")
        print(f"사용 가능한 GPU: {gpu_ids}")
        print(f"분산 학습 사용: {use_distributed}")
        
        if use_distributed:
            print(f"분산 학습이 활성화됩니다 ({len(gpu_ids)}개 GPU)")
        else:
            print("단일 GPU 또는 CPU 학습입니다")
            
    except Exception as e:
        print(f"설정 테스트 실패: {e}")

def test_model_creation():
    """모델 생성 테스트"""
    print("\n=== 모델 생성 테스트 ===")
    
    try:
        from models.yolo_mc import YOLOWithMCDropout
        from utils.distributed_utils import setup_model_for_distributed
        
        # 테스트용 모델 생성
        model = YOLOWithMCDropout(
            model_name="yolov8m",
            dropout_rate=0.1,
            feature_alignment_enabled=True,
            num_classes=91,
            ema_decay=0.999
        )
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        use_distributed = torch.cuda.device_count() > 1
        
        # 분산 학습용 모델 설정
        model = setup_model_for_distributed(
            model, 
            device, 
            use_distributed,
            find_unused_parameters=True
        )
        
        print(f"모델 생성 성공: {type(model)}")
        print(f"디바이스: {device}")
        print(f"분산 학습: {use_distributed}")
        
        if use_distributed:
            print("모델이 DistributedDataParallel로 래핑되었습니다")
        
    except Exception as e:
        print(f"모델 생성 테스트 실패: {e}")

def main():
    """메인 테스트 함수"""
    print("SSOD YOLO MC 분산 학습 테스트")
    print("=" * 50)
    
    test_gpu_availability()
    test_distributed_config()
    test_model_creation()
    
    print("\n=== 추천 실행 명령어 ===")
    
    gpu_count = torch.cuda.device_count()
    
    if gpu_count >= 4:
        print("4장 GPU 분산 학습:")
        print("cd src && uv run train.py --use_distributed --gpu_ids 0,1,2,3")
    elif gpu_count >= 2:
        print(f"{gpu_count}장 GPU 분산 학습:")
        print(f"cd src && uv run train.py --use_distributed --gpu_ids {','.join(map(str, range(gpu_count)))}")
    elif gpu_count == 1:
        print("단일 GPU 학습:")
        print("cd src && uv run train.py --gpu_ids 0")
    else:
        print("CPU 학습:")
        print("cd src && uv run train.py --device cpu")

if __name__ == "__main__":
    main() 