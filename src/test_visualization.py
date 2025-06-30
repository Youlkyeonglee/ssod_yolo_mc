#!/usr/bin/env python3

import torch
import numpy as np
from pathlib import Path
from utils.visualization import visualize_batch
from data.semi_supervised_dataset import SemiSupervisedDataset
from train import get_transform
import yaml

def test_visualization():
    """시각화 함수 테스트"""
    
    # 설정 로드
    config_path = "configs/yolo_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 데이터셋 초기화
    transform = get_transform(train=False, img_size=config['data']['img_size'])
    
    dataset_manager = SemiSupervisedDataset(
        config_path=config_path,
        percent=10.0,
        seed=1,
        transform=transform,
        max_samples=20  # 테스트용 적은 샘플
    )
    
    # 데이터 로더 생성
    _, _, val_loader = dataset_manager.get_dataloaders(batch_size=4, num_workers=0)
    
    # 첫 번째 배치 가져오기
    sample_batch = next(iter(val_loader))
    
    print("=== 배치 정보 ===")
    print(f"배치 키: {sample_batch.keys()}")
    
    images = sample_batch['images']
    targets = sample_batch['labels']
    img_paths = sample_batch['img_paths']
    
    print(f"배치 내 이미지 수: {len(images)}")
    print(f"이미지 경로: {img_paths[0]}")
    print(f"이미지 이름: {Path(img_paths[0]).name}")
    print(f"이미지 텐서 크기: {images.shape}")
    
    print(f"\n=== 타겟 정보 ===")
    print(f"배치 내 타겟 리스트 길이: {len(targets)}")  # 이것이 이미지 수
    
    # 첫 번째 (그리고 유일한) 이미지의 타겟 정보
    if len(targets) > 0 and targets[0] is not None:
        target_tensor = targets[0]  # 첫 번째 이미지의 타겟들
        print(f"첫 번째 이미지의 객체 수: {len(target_tensor)}")
        print(f"타겟 텐서 크기: {target_tensor.shape}")
        
        if len(target_tensor) > 0:
            print(f"\n=== 각 객체 정보 ===")
            class_names = config['data']['names']
            for i, target in enumerate(target_tensor):
                class_id = int(target[0])
                class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown({class_id})"
                bbox = target[1:5]  # x_center, y_center, width, height (정규화된 좌표)
                print(f"  객체 {i+1}: {class_name} (ID: {class_id})")
                print(f"    bbox (정규화): x_center={bbox[0]:.3f}, y_center={bbox[1]:.3f}, width={bbox[2]:.3f}, height={bbox[3]:.3f}")
        else:
            print("이 이미지에는 객체가 없습니다 (배경 이미지)")
    else:
        print("타겟 정보가 없습니다")
    
    # 빈 예측 생성 (테스트용)
    predictions = [None] * len(images)
    
    # 시각화 실행
    save_path = Path("test_visualization.png")
    visualize_batch(
        images=images,
        targets=targets,
        predictions=predictions,
        class_names=val_loader.dataset.class_names,
        save_path=save_path,
        max_images=4
    )
    
    print(f"\n=== 시각화 완료 ===")
    print(f"시각화 이미지 저장: {save_path}")

if __name__ == "__main__":
    test_visualization() 