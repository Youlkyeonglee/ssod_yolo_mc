#!/usr/bin/env python3
"""간단한 GT 데이터 시각화 테스트 스크립트"""

import sys
import yaml
from pathlib import Path
import torch
import torchvision.transforms as transforms
from data.semi_supervised_dataset import SemiSupervisedDataset
from utils.visualization import visualize_gt_data, visualize_dataset_statistics

def denormalize(img, mean, std):
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

def main():
    print("🔍 간단한 GT 데이터 시각화 테스트 시작...")
    
    # 설정 파일 경로
    config_path = "configs/yolo_config.yaml"
    
    # 테스트용 파라미터 (매우 작은 데이터로 테스트)
    labeled_ratio = 1.0  # 1%만 사용
    seed = 1
    max_samples = 8  # 매우 적은 수
    
    # 간단한 변환 함수
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    try:
        # 데이터셋 관리자 초기화
        print("📊 데이터셋 초기화 중...")
        dataset_manager = SemiSupervisedDataset(
            config_path=config_path,
            percent=labeled_ratio,
            seed=seed,
            transform=transform,
            max_samples=max_samples
        )
        
        # 데이터 로더 생성 (매우 작은 배치)
        labeled_loader, unlabeled_loader, val_loader = dataset_manager.get_dataloaders(
            batch_size=2,  # 매우 작은 배치
            num_workers=0  # 단일 프로세스
        )
        
        print(f"✅ Labeled: {len(labeled_loader.dataset)} 샘플")
        print(f"✅ Unlabeled: {len(unlabeled_loader.dataset)} 샘플") 
        print(f"✅ Validation: {len(val_loader.dataset)} 샘플")
        
        # COCO 클래스 이름
        coco_class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # 결과 저장 디렉토리
        save_dir = Path("test_gt_visualization_simple")
        save_dir.mkdir(exist_ok=True)
        
        # 1. Labeled GT 데이터 시각화
        print("\n📊 Labeled GT 데이터 시각화 중...")
        labeled_gt_dir = save_dir / "labeled"
        visualize_gt_data(
            data_loader=labeled_loader,
            class_names=coco_class_names,
            save_dir=labeled_gt_dir,
            data_type="labeled",
            max_batches=1,  # 1개 배치만
            max_images_per_batch=2  # 2개 이미지만
        )
        
        # 2. Validation GT 데이터 시각화
        print("\n📊 Validation GT 데이터 시각화 중...")
        val_gt_dir = save_dir / "validation"
        visualize_gt_data(
            data_loader=val_loader,
            class_names=coco_class_names,
            save_dir=val_gt_dir,
            data_type="validation",
            max_batches=1,  # 1개 배치만
            max_images_per_batch=2  # 2개 이미지만
        )
        
        # 3. 데이터셋 통계 시각화
        print("\n📊 데이터셋 통계 시각화 중...")
        visualize_dataset_statistics(
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            val_loader=val_loader,
            class_names=coco_class_names,
            save_dir=save_dir
        )
        
        print(f"\n🎉 시각화 완료! 결과는 {save_dir}에 저장되었습니다.")
        
        # 생성된 파일 목록 출력
        print("\n📁 생성된 파일:")
        for file_path in save_dir.rglob('*.png'):
            print(f"  - {file_path.relative_to(save_dir)}")
        for file_path in save_dir.rglob('*.txt'):
            print(f"  - {file_path.relative_to(save_dir)}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 