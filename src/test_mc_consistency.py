#!/usr/bin/env python3
"""
MC Dropout Consistency Loss 테스트 스크립트
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.yolo_losses import create_mc_consistency_loss
from uncertainty.mc_dropout import MCDropoutDetector
from models.yolo_mc import YOLOWithMCDropout
import yaml

def test_mc_consistency_loss():
    """MC Consistency Loss 테스트"""
    print("🧪 MC Dropout Consistency Loss 테스트 시작")
    
    # 설정 파일 로드
    config_path = project_root / "configs" / "yolo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 사용 디바이스: {device}")
    
    # MC Consistency Loss 생성
    mc_consistency_loss_fn = create_mc_consistency_loss(
        alpha=config['training']['semi_supervised'].get('mc_consistency_alpha', 1.0),
        beta=config['training']['semi_supervised'].get('mc_consistency_beta', 0.5),
        temperature=config['training']['semi_supervised'].get('mc_consistency_temperature', 1.0),
        uncertainty_threshold=config['training']['semi_supervised'].get('mc_uncertainty_threshold', 0.1),
        use_adaptive_weighting=config['training']['semi_supervised'].get('mc_use_adaptive_weighting', True)
    ).to(device)
    
    print("✅ MC Consistency Loss 생성 완료")
    
    # 테스트 데이터 생성
    batch_size = 2
    num_classes = 80
    img_size = 640
    
    # Strong augmentation된 이미지 시뮬레이션
    strong_unlabeled_images = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    # High quality pseudo labels 시뮬레이션
    high_quality_pseudo_labels = []
    for i in range(batch_size):
        # 각 배치에 대해 몇 개의 pseudo label 생성
        num_objects = torch.randint(1, 5, (1,)).item()
        pseudo_boxes = torch.rand(num_objects, 5).to(device)  # [class_id, x, y, w, h]
        pseudo_boxes[:, 0] = torch.randint(0, num_classes, (num_objects,)).float()  # class_id
        pseudo_boxes[:, 1:5] = torch.clamp(pseudo_boxes[:, 1:5], 0.1, 0.9)  # bbox coordinates
        
        high_quality_pseudo_labels.append({
            'boxes': pseudo_boxes,
            'image_path': f'test_image_{i}.jpg',
            'uncertainty_stats': {
                'mc_samples': 5,
                'detections_count': num_objects,
                'reliability_score': 0.8,
                'quality_grade': 'high'
            }
        })
    
    print(f"✅ 테스트 데이터 생성 완료: {batch_size} 배치, {len(high_quality_pseudo_labels)} pseudo labels")
    
    # YOLO 모델 생성 (테스트용)
    model = YOLOWithMCDropout(
        model_name="yolov8n",  # 작은 모델로 테스트
        dropout_rate=0.1,
        feature_alignment_enabled=False,
        num_classes=num_classes,
        ema_decay=0.999
    ).to(device)
    
    # MC Dropout Detector 생성
    detector = MCDropoutDetector(
        model=model,
        num_samples=3,  # 테스트용으로 적은 샘플 수
        dropout_rate=0.1,
        box_std_threshold=0.1,
        entropy_threshold=0.5,
        conf_threshold=0.25
    )
    
    print("✅ YOLO 모델 및 MC Dropout Detector 생성 완료")
    
    # Student 모델을 detector에 설정하여 MC 예측 수행
    student_model = model.student_model
    original_model = detector.model
    detector.model = student_model.model
    
    try:
        # detector.predict_with_uncertainty 사용하여 Student MC Dropout 예측
        student_mc_result = detector.predict_with_uncertainty(
            strong_unlabeled_images, 
            device=device, 
            config=config
        )
        
        print(f"✅ Student MC Dropout 예측 완료: {len(student_mc_result) if student_mc_result else 0} 결과")
        
        # MC Consistency Loss 계산
        if student_mc_result and len(student_mc_result) > 0:
            mc_consistency_result = mc_consistency_loss_fn(
                student_mc_predictions=student_mc_result,
                high_quality_pseudo_labels=high_quality_pseudo_labels,
                strong_unlabeled_images=strong_unlabeled_images,
                return_components=True
            )
            
            print("✅ MC Consistency Loss 계산 완료")
            print(f"   - Total Loss: {mc_consistency_result['total'].item():.6f}")
            print(f"   - Pseudo Consistency: {mc_consistency_result['pseudo_consistency'].item():.6f}")
            print(f"   - MC Uncertainty: {mc_consistency_result['mc_uncertainty'].item():.6f}")
            
            if 'adaptive_weights' in mc_consistency_result:
                print(f"   - Adaptive Alpha: {mc_consistency_result['adaptive_weights']['alpha']:.3f}")
                print(f"   - Adaptive Beta: {mc_consistency_result['adaptive_weights']['beta']:.3f}")
        else:
            print("⚠️  Student MC 결과가 없어서 MC Consistency Loss 계산을 건너뜁니다.")
            
    except Exception as e:
        print(f"❌ MC Consistency Loss 계산 중 오류: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 원래 모델 복원
        detector.model = original_model
    
    print("🎉 MC Dropout Consistency Loss 테스트 완료!")

if __name__ == "__main__":
    test_mc_consistency_loss() 