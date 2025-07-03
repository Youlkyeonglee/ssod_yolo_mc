#!/usr/bin/env python3
"""새로운 평가 함수 테스트 스크립트"""

import sys
import yaml
from pathlib import Path
import torch
import torchvision.transforms as transforms
from data.semi_supervised_dataset import SemiSupervisedDataset
from models.yolo_mc import YOLOWithMCDropout

def test_evaluation():
    print("🔍 새로운 평가 함수 테스트 시작...")
    
    # 설정 파일 경로
    config_path = "configs/yolo_config.yaml"
    
    # 테스트용 파라미터
    labeled_ratio = 1.0
    seed = 1
    max_samples = 16
    
    # 변환 함수
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    try:
        # 설정 로드
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # 데이터셋 초기화
        print("📊 데이터셋 초기화 중...")
        dataset_manager = SemiSupervisedDataset(
            config_path=config_path,
            percent=labeled_ratio,
            seed=seed,
            transform=transform,
            max_samples=max_samples
        )
        
        # 데이터 로더 생성
        labeled_loader, unlabeled_loader, val_loader = dataset_manager.get_dataloaders(
            batch_size=2,
            num_workers=0
        )
        
        print(f"✅ Validation: {len(val_loader.dataset)} 샘플")
        
        # 모델 생성
        print("🤖 모델 초기화 중...")
        model = YOLOWithMCDropout(
            model_name=config['model']['name'],
            dropout_rate=config['model']['dropout']['rate'],
            feature_alignment_enabled=config['model']['feature_alignment']['enabled'],
            num_classes=config['data']['nc'],
            ema_decay=config.get('model', {}).get('ema', {}).get('decay', 0.999)
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        print(f"✅ 모델 생성 완료 (디바이스: {device})")
        
        # 평가 함수 테스트
        print("\n🔍 평가 함수 테스트 중...")
        
        # train.py에서 정의된 평가 함수 import
        from train import evaluate_model_wrapper
        
        mAP50, mAP50_95 = evaluate_model_wrapper(
            model=model,
            val_loader=val_loader,
            device=device,
            epoch=0,
            save_dir=Path("."),
            val_data_path="configs/coco_val.yaml"
        )
        
        print(f"\n📊 평가 결과:")
        print(f"  - mAP@0.5: {mAP50:.4f}")
        print(f"  - mAP@0.5:0.95: {mAP50_95:.4f}")
        
        # 결과 검증
        if mAP50 > 0.0 or mAP50_95 > 0.0:
            print("✅ 평가 함수가 정상적으로 작동합니다!")
        else:
            print("⚠️  평가 결과가 0입니다. 모델이 아직 학습되지 않았거나 평가에 문제가 있을 수 있습니다.")
        
        return 0
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_evaluation()) 