import os
import torch
from typing import List, Optional, Tuple, Dict, Any
import logging
from pathlib import Path


def is_gpu_available() -> bool:
    """GPU 사용 가능 여부 확인"""
    return torch.cuda.is_available()


def get_available_gpus() -> List[int]:
    """사용 가능한 GPU ID 리스트 반환 (사용 중인 GPU 제외)"""
    if not is_gpu_available():
        return []
    
    available_gpus = []
    total_gpus = torch.cuda.device_count()
    
    for gpu_id in range(total_gpus):
        try:
            # GPU 메모리 사용량 확인
            memory_allocated = torch.cuda.memory_allocated(gpu_id)
            memory_reserved = torch.cuda.memory_reserved(gpu_id)
            
            # GPU가 사용 중인지 확인 (메모리 사용량이 일정 임계값 이상이면 사용 중으로 간주)
            # 1GB 이상 사용 중이면 제외
            if memory_allocated < 1024 * 1024 * 1024:  # 1GB
                available_gpus.append(gpu_id)
            else:
                print(f"⚠️  GPU {gpu_id}는 사용 중입니다 (메모리 사용량: {memory_allocated / 1024**3:.2f}GB)")
        except Exception as e:
            print(f"⚠️  GPU {gpu_id} 상태 확인 실패: {e}")
            # 확인 실패 시 일단 포함
            available_gpus.append(gpu_id)
    
    print(f"✅ 사용 가능한 GPU: {available_gpus}")
    return available_gpus


def setup_gpu_config(config: Dict[str, Any]) -> Tuple[str, List[int], bool]:
    """
    GPU 설정 구성
    
    Args:
        config: 설정 딕셔너리
    
    Returns:
        device: 주요 디바이스 ('cpu' 또는 'cuda:0')
        gpu_ids: 사용할 GPU ID 리스트
        use_distributed: 분산 학습 사용 여부 (항상 False)
    """
    gpu_config = config.get('gpu', {})
    
    # GPU 사용 여부 확인
    use_gpu = gpu_config.get('use_gpu', True) and is_gpu_available()
    
    if not use_gpu:
        return 'cpu', [], False
    
    # GPU ID 설정
    if gpu_config.get('auto_detect', True):
        gpu_ids = get_available_gpus()
    else:
        specified_ids = gpu_config.get('gpu_ids', [0])
        available_ids = get_available_gpus()
        gpu_ids = [gpu_id for gpu_id in specified_ids if gpu_id in available_ids]
    
    if not gpu_ids:
        return 'cpu', [], False
    
    # 주요 디바이스 설정
    device = f'cuda:{gpu_ids[0]}'
    
    # 분산 학습은 사용하지 않음
    return device, gpu_ids, False


def get_gpu_memory_info() -> Dict[int, Dict[str, float]]:
    """GPU 메모리 정보 반환"""
    if not is_gpu_available():
        return {}
    
    memory_info = {}
    for gpu_id in range(torch.cuda.device_count()):
        try:
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3    # GB
            total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3  # GB
            
            memory_info[gpu_id] = {
                'allocated': allocated,
                'reserved': reserved,
                'total': total,
                'free': total - reserved
            }
        except Exception as e:
            print(f"⚠️  GPU {gpu_id} 메모리 정보 확인 실패: {e}")
    
    return memory_info 