#!/usr/bin/env python3
"""
Unlabeled 데이터 로딩 테스트
"""

import torch
from data.semi_supervised_dataset import SemiSupervisedDataset
from torchvision import transforms
import yaml

def test_unlabeled_fix():
    """Unlabeled 데이터 로딩 테스트"""
    print("🧪 Unlabeled 데이터 로딩 테스트 시작")
    
    try:
        # 기본 transform 생성
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        
        # 데이터셋 관리자 초기화
        print("🔧 데이터셋 초기화 중...")
        dataset_manager = SemiSupervisedDataset(
            config_path='configs/yolo_config_test.yaml',
            percent=10.0,
            seed=1,
            transform=transform,
            max_samples=5  # 테스트용으로 적은 수
        )
        
        print(f"✅ 데이터셋 초기화 완료")
        print(f"  - Labeled: {len(dataset_manager.labeled_dataset)} 샘플")
        print(f"  - Unlabeled: {len(dataset_manager.unlabeled_dataset)} 샘플")
        print(f"  - Validation: {len(dataset_manager.val_dataset)} 샘플")
        
        # 데이터 로더 생성
        labeled_loader, unlabeled_loader, val_loader = dataset_manager.get_dataloaders(
            batch_size=2,
            num_workers=0
        )
        
        print(f"\n✅ 데이터 로더 생성 완료")
        
        # Unlabeled 배치 테스트
        print("\n🔍 Unlabeled 배치 테스트...")
        unlabeled_batch = next(iter(unlabeled_loader))
        print(f"📊 Unlabeled 배치 키들: {list(unlabeled_batch.keys())}")
        print(f"  - images: {unlabeled_batch['images'].shape}")
        print(f"  - labels: {len(unlabeled_batch['labels'])}개")
        print(f"  - paths: {len(unlabeled_batch.get('paths', []))}개")
        
        if 'paths' in unlabeled_batch:
            print(f"  ✅ paths 키 존재!")
            for i, path in enumerate(unlabeled_batch['paths']):
                print(f"    경로 {i}: {path}")
        else:
            print(f"  ❌ paths 키 없음!")
        
        print(f"\n✅ Unlabeled 데이터 로딩 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_unlabeled_fix() 