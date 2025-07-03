#!/usr/bin/env python3
"""GT 데이터 시각화 테스트 스크립트"""

import sys
import yaml
from pathlib import Path
import torch
import torchvision.transforms as transforms
from data.semi_supervised_dataset import SemiSupervisedDataset
from utils.visualization import visualize_gt_data

def get_transform(train: bool = True, img_size: int = 640):
    """데이터 변환 함수 생성 (PyTorch transforms 사용)"""
    if train:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform

def main():
    print("GT 데이터 시각화 테스트 시작...")
    
    # 설정 파일 경로
    config_path = "configs/yolo_config.yaml"
    
    # 테스트용 파라미터
    labeled_ratio = 10.0
    seed = 1
    max_samples = 16  # 테스트용으로 적은 수
    
    # 변환 함수 생성
    transform = get_transform(train=True, img_size=640)
    
    try:
        # 데이터셋 관리자 초기화
        print("데이터셋 초기화 중...")
        dataset_manager = SemiSupervisedDataset(
            config_path=config_path,
            percent=labeled_ratio,
            seed=seed,
            transform=transform,
            max_samples=max_samples
        )
        
        # 데이터 로더 생성
        labeled_loader, unlabeled_loader, val_loader = dataset_manager.get_dataloaders(
            batch_size=4,
            num_workers=2
        )
        
        print(f"Labeled: {len(labeled_loader.dataset)} 샘플")
        print(f"Unlabeled: {len(unlabeled_loader.dataset)} 샘플") 
        print(f"Validation: {len(val_loader.dataset)} 샘플")
        
        # 클래스 이름 가져오기
        class_names = dataset_manager.class_names
        print(f"클래스 수: {len(class_names)}")
        
        # 결과 저장 디렉토리
        save_dir = Path("test_gt_visualization")
        save_dir.mkdir(exist_ok=True)
        
        # Train GT 데이터 시각화
        print("\nTrain GT 데이터 시각화 중...")
        train_gt_dir = save_dir / "train"
        visualize_gt_data(
            data_loader=labeled_loader,
            class_names=class_names,
            save_dir=train_gt_dir,
            data_type="train",
            max_batches=2,
            max_images_per_batch=4
        )
        
        # Val GT 데이터 시각화
        print("\nValidation GT 데이터 시각화 중...")
        val_gt_dir = save_dir / "val"
        visualize_gt_data(
            data_loader=val_loader,
            class_names=class_names,
            save_dir=val_gt_dir,
            data_type="val",
            max_batches=1,
            max_images_per_batch=4
        )
        
        print(f"\n✓ 시각화 완료! 결과는 {save_dir}에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 