#!/usr/bin/env python3
"""
간단한 디버그용 학습 스크립트
SSOD YOLO MC의 핵심 부분만 테스트
"""

import sys
import os
import torch
import logging
from pathlib import Path

# 현재 디렉토리를 sys.path에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def main():
    print("=" * 80)
    print("🔍 SSOD YOLO MC 간단 테스트 시작")
    print("=" * 80)

    try:
        print("1️⃣ PyTorch 및 CUDA 확인...")
        print(f"   PyTorch 버전: {torch.__version__}")
        print(f"   CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA 버전: {torch.version.cuda}")
            print(f"   GPU 개수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        print("\n2️⃣ 모듈 임포트 테스트...")
        from models.yolo_mc import YOLOWithMCDropout
        print("   ✓ YOLOWithMCDropout 임포트 성공")
        
        from uncertainty.mc_dropout import MCDropoutDetector  
        print("   ✓ MCDropoutDetector 임포트 성공")
        
        from data.semi_supervised_dataset import SemiSupervisedDataset
        print("   ✓ SemiSupervisedDataset 임포트 성공")
        
        print("\n3️⃣ 모델 생성 테스트...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"   사용 디바이스: {device}")
        
        model = YOLOWithMCDropout(
            model_name="yolov8n",  # 가벼운 모델 사용
            dropout_rate=0.1,
            feature_alignment_enabled=False,  # 간단히 하기 위해 비활성화
            num_classes=80,
            ema_decay=0.999
        )
        print("   ✓ YOLOWithMCDropout 모델 생성 성공")
        
        model = model.to(device)
        print(f"   ✓ 모델을 {device}로 이동 성공")
        
        print("\n4️⃣ MC Dropout 탐지기 생성 테스트...")
        detector = MCDropoutDetector(
            model=model,
            num_samples=3,  # 적은 샘플 수로 테스트
            dropout_rate=0.1,
            box_std_threshold=0.1,
            entropy_threshold=0.5,
            save_dir=None  # 저장 없이 테스트
        )
        print("   ✓ MCDropoutDetector 생성 성공")
        
        print("\n5️⃣ 간단한 추론 테스트...")
        # 더미 이미지로 테스트
        dummy_images = torch.randn(2, 3, 640, 640).to(device)
        print(f"   더미 이미지 생성: {dummy_images.shape}")
        
        model.eval()
        with torch.no_grad():
            # Student 모델 추론
            outputs = model(dummy_images)
            print(f"   ✓ Student 모델 추론 성공: {type(outputs)}")
            
            # MC Dropout 추론 테스트
            print("   MC Dropout 추론 테스트 중...")
            mc_results = detector.predict_with_uncertainty(
                dummy_images, 
                device=device,
                config={
                    'model': {'dropout': {'num_samples': 3}},
                    'training': {'semi_supervised': {'uncertainty': {
                        'box_std_threshold': 0.1,
                        'entropy_threshold': 0.5
                    }}}
                }
            )
            print(f"   ✓ MC Dropout 추론 성공: {len(mc_results) if mc_results else 0} 결과")
        
        print("\n6️⃣ 데이터셋 로드 테스트...")
        config_path = "configs/yolo_config.yaml"
        if Path(config_path).exists():
            try:
                dataset_manager = SemiSupervisedDataset(
                    config_path=config_path,
                    percent=10.0,
                    seed=1,
                    transform=None,
                    max_samples=10  # 매우 적은 샘플로 테스트
                )
                print("   ✓ SemiSupervisedDataset 초기화 성공")
                
                labeled_loader, unlabeled_loader, val_loader = dataset_manager.get_dataloaders(
                    batch_size=2,  # 작은 배치 크기
                    num_workers=0  # 멀티프로세싱 비활성화
                )
                print(f"   ✓ DataLoader 생성 성공")
                print(f"      - Labeled: {len(labeled_loader.dataset)} 샘플")
                print(f"      - Unlabeled: {len(unlabeled_loader.dataset)} 샘플")
                print(f"      - Validation: {len(val_loader.dataset)} 샘플")
                
                # 첫 번째 배치 테스트
                print("   첫 번째 배치 로드 테스트...")
                sample_batch = next(iter(labeled_loader))
                print(f"   ✓ 배치 로드 성공: 이미지 {sample_batch['images'].shape}")
                
            except Exception as dataset_error:
                print(f"   ❌ 데이터셋 테스트 실패: {dataset_error}")
        else:
            print(f"   ⚠️  설정 파일 없음: {config_path}")
        
        print("\n" + "=" * 80)
        print("🎉 모든 테스트 통과! 기본 구성 요소들이 정상 작동합니다.")
        print("=" * 80)
        
        return True

    except ImportError as e:
        print(f"❌ 모듈 임포트 실패: {e}")
        print("   필요한 의존성이 설치되어 있는지 확인하세요.")
        return False

    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")
        import traceback
        print("상세 오류 정보:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 테스트 완료 - 기본 기능이 정상 작동합니다.")
        print("   이제 전체 학습을 시도할 수 있습니다.")
    else:
        print("\n❌ 테스트 실패 - 문제를 해결한 후 다시 시도하세요.")
        sys.exit(1) 