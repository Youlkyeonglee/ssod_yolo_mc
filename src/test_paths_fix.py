#!/usr/bin/env python3
"""
paths 키 전달 테스트
"""

import torch
from data.semi_supervised_dataset import SemiSupervisedDataset
import yaml

def test_paths_fix():
    """paths 키가 제대로 전달되는지 테스트"""
    print("🧪 paths 키 전달 테스트 시작")
    
    # 설정 파일 로드
    with open('configs/yolo_config_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        # 기본 transform 생성
        from torchvision import transforms
        
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
        
        # 데이터 로더 생성
        labeled_loader, unlabeled_loader, val_loader = dataset_manager.get_dataloaders(
            batch_size=2,
            num_workers=0
        )
        
        print(f"✅ 데이터 로더 생성 완료")
        print(f"  - Labeled: {len(labeled_loader.dataset)} 샘플")
        print(f"  - Unlabeled: {len(unlabeled_loader.dataset)} 샘플")
        
        # 첫 번째 배치 테스트
        print("\n🔍 첫 번째 배치 테스트...")
        
        # Labeled 배치 테스트
        labeled_batch = next(iter(labeled_loader))
        print(f"📊 Labeled 배치 키들: {list(labeled_batch.keys())}")
        print(f"  - images: {labeled_batch['images'].shape}")
        print(f"  - labels: {len(labeled_batch['labels'])}개")
        print(f"  - paths: {len(labeled_batch.get('paths', []))}개")
        
        if 'paths' in labeled_batch:
            print(f"  ✅ paths 키 존재!")
            for i, path in enumerate(labeled_batch['paths']):
                print(f"    경로 {i}: {path}")
        else:
            print(f"  ❌ paths 키 없음!")
        
        # Unlabeled 배치 테스트
        unlabeled_batch = next(iter(unlabeled_loader))
        print(f"\n📊 Unlabeled 배치 키들: {list(unlabeled_batch.keys())}")
        print(f"  - images: {unlabeled_batch['images'].shape}")
        print(f"  - labels: {len(unlabeled_batch['labels'])}개")
        print(f"  - paths: {len(unlabeled_batch.get('paths', []))}개")
        
        if 'paths' in unlabeled_batch:
            print(f"  ✅ paths 키 존재!")
            for i, path in enumerate(unlabeled_batch['paths']):
                print(f"    경로 {i}: {path}")
        else:
            print(f"  ❌ paths 키 없음!")
        
        print(f"\n✅ paths 키 전달 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_paths_fix() 