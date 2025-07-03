#!/usr/bin/env python3
"""
실제 MC Dropout 결과에서 박스 좌표 유효성 검사 테스트
"""

import torch
import numpy as np
from models.yolo_mc import YOLOWithMCDropout
from uncertainty.mc_dropout import MCDropoutDetector
import yaml

def test_mc_dropout_box_validation():
    """MC Dropout 결과에서 박스 좌표 유효성 검사 테스트"""
    print("🧪 MC Dropout 박스 좌표 유효성 검사 테스트 시작")
    
    # 설정 파일 로드
    with open('configs/yolo_config_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 디바이스 설정
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 디바이스: {device}")
    
    # YOLO 모델 생성
    print("🔧 YOLO 모델 생성 중...")
    model = YOLOWithMCDropout(
        model_name=config['model']['name'],
        dropout_rate=config['model']['dropout']['rate'],
        feature_alignment_enabled=config['model']['feature_alignment']['enabled'],
        num_classes=config['data']['nc'],
        ema_decay=config['model']['ema']['decay']
    ).to(device)
    
    # MC Dropout Detector 생성
    print("🔧 MC Dropout Detector 생성 중...")
    detector = MCDropoutDetector(
        model=model.student_model,
        num_samples=config['model']['dropout']['num_samples'],
        dropout_rate=config['model']['dropout']['rate'],
        box_std_threshold=config['training']['semi_supervised']['uncertainty']['box_std_threshold'],
        entropy_threshold=config['training']['semi_supervised']['uncertainty']['entropy_threshold'],
        conf_threshold=config['training']['semi_supervised']['conf_threshold']
    )
    
    # 테스트 이미지 생성 (더미 데이터)
    print("🔧 테스트 이미지 생성 중...")
    batch_size = 2
    img_size = config['data']['img_size']
    test_images = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    print(f"📊 테스트 이미지 크기: {test_images.shape}")
    
    # MC Dropout 예측 수행
    print("🔍 MC Dropout 예측 수행 중...")
    try:
        mc_results = detector.predict_with_uncertainty(
            test_images, 
            device=device, 
            save_predictions=False,
            config=config
        )
        
        if mc_results:
            print(f"✅ MC Dropout 예측 성공: {len(mc_results)} 배치 결과")
            
            # 각 배치 결과 검사
            for batch_idx, batch_result in enumerate(mc_results):
                print(f"\n📊 배치 {batch_idx + 1} 결과:")
                print(f"  - 박스 수: {len(batch_result.get('boxes', []))}")
                print(f"  - 점수 수: {len(batch_result.get('scores', []))}")
                print(f"  - 레이블 수: {len(batch_result.get('labels', []))}")
                
                if len(batch_result.get('boxes', [])) > 0:
                    boxes = batch_result['boxes']
                    print(f"  - 박스 좌표 범위:")
                    print(f"    x: [{boxes[:, 0].min().item():.4f}, {boxes[:, 0].max().item():.4f}]")
                    print(f"    y: [{boxes[:, 1].min().item():.4f}, {boxes[:, 1].max().item():.4f}]")
                    print(f"    w: [{boxes[:, 2].min().item():.4f}, {boxes[:, 2].max().item():.4f}]")
                    print(f"    h: [{boxes[:, 3].min().item():.4f}, {boxes[:, 3].max().item():.4f}]")
                    
                    # 유효성 검사
                    x_valid = torch.all((boxes[:, 0] >= 0) & (boxes[:, 0] <= 1))
                    y_valid = torch.all((boxes[:, 1] >= 0) & (boxes[:, 1] <= 1))
                    w_valid = torch.all((boxes[:, 2] >= 0.01) & (boxes[:, 2] <= 1))
                    h_valid = torch.all((boxes[:, 3] >= 0.01) & (boxes[:, 3] <= 1))
                    
                    print(f"  - 유효성 검사 결과:")
                    print(f"    x 좌표 유효: {x_valid}")
                    print(f"    y 좌표 유효: {y_valid}")
                    print(f"    width 유효: {w_valid}")
                    print(f"    height 유효: {h_valid}")
                    
                    if x_valid and y_valid and w_valid and h_valid:
                        print(f"  ✅ 모든 박스 좌표가 유효합니다!")
                    else:
                        print(f"  ❌ 일부 박스 좌표가 유효하지 않습니다.")
                        
                        # 문제가 있는 박스들 출력
                        invalid_mask = ~((boxes[:, 0] >= 0) & (boxes[:, 0] <= 1) & 
                                       (boxes[:, 1] >= 0) & (boxes[:, 1] <= 1) & 
                                       (boxes[:, 2] >= 0.01) & (boxes[:, 2] <= 1) & 
                                       (boxes[:, 3] >= 0.01) & (boxes[:, 3] <= 1))
                        
                        if invalid_mask.any():
                            print(f"  🔍 유효하지 않은 박스들:")
                            for i, is_invalid in enumerate(invalid_mask):
                                if is_invalid:
                                    print(f"    박스 {i}: {boxes[i]}")
                else:
                    print(f"  ⚠️  이 배치에는 박스가 없습니다.")
        else:
            print("❌ MC Dropout 예측 결과가 없습니다.")
            
    except Exception as e:
        print(f"❌ MC Dropout 예측 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ MC Dropout 박스 좌표 유효성 검사 테스트 완료")

if __name__ == "__main__":
    test_mc_dropout_box_validation() 