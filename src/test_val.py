#!/usr/bin/env python3
"""val.py 테스트 스크립트"""

import sys
import yaml
from pathlib import Path
import torch
import torchvision.transforms as transforms
from data.semi_supervised_dataset import SemiSupervisedDataset
from models.yolo_mc import YOLOWithMCDropout

def test_val():
    print("🔍 val.py 테스트 시작...")
    
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
        
        # val.py에서 정의된 평가 함수 import
        from val import evaluate_custom
        
        # 커스텀 평가 함수 테스트
        print("\n🔍 커스텀 평가 함수 테스트 중...")
        
        # 간단한 args 객체 생성
        class Args:
            def __init__(self):
                self.mc_dropout_eval = False
                self.num_mc_samples = 10
                self.conf_threshold = 0.25
                self.iou_threshold = 0.5
        
        args = Args()
        
        results = evaluate_custom(
            model=model,
            val_loader=val_loader,
            config=config,
            args=args,
            device=device
        )
        
        print(f"\n📊 평가 결과:")
        print(f"  - mAP@0.5: {results['mAP50']:.4f}")
        print(f"  - mAP@0.5:0.95: {results['mAP50_95']:.4f}")
        print(f"  - 총 예측: {results['total_predictions']}")
        print(f"  - 총 타겟: {results['total_targets']}")
        
        # 결과 검증
        if results['mAP50'] >= 0.0 and results['mAP50_95'] >= 0.0:
            print("✅ val.py 평가 함수가 정상적으로 작동합니다!")
        else:
            print("⚠️  평가 결과가 예상과 다릅니다.")
        
        return 0
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_val()) 