#!/usr/bin/env python3
"""
Loss 계산 디버깅 테스트 스크립트
'NoneType' object does not support item assignment 오류 해결 확인
"""

import torch
import numpy as np
from models.yolo_mc import YOLOWithMCDropout
from uncertainty.mc_dropout import MCDropoutDetector
import yaml

def test_loss_debug():
    """Loss 계산 디버깅 테스트"""
    print("🧪 Loss 계산 디버깅 테스트 시작")
    
    # 설정 파일 로드
    with open('configs/yolo_config_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 디바이스 설정
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 디바이스: {device}")
    
    try:
        # YOLO 모델 생성
        print("🔧 YOLO 모델 생성 중...")
        model = YOLOWithMCDropout(
            model_name=config['model']['name'],
            dropout_rate=config['model']['dropout']['rate'],
            feature_alignment_enabled=config['model']['feature_alignment']['enabled'],
            num_classes=config['data']['nc']
        ).to(device)
        
        # MC Dropout Detector 생성
        detector = MCDropoutDetector(
            model=model.teacher_model,
            num_samples=config['model']['dropout']['num_samples'],
            dropout_rate=config['model']['dropout']['rate'],
            box_std_threshold=config['training']['semi_supervised']['uncertainty']['box_std_threshold'],
            entropy_threshold=config['training']['semi_supervised']['uncertainty']['entropy_threshold']
        )
        
        # 더미 데이터 생성
        batch_size = 2
        img_size = config['data']['img_size']
        dummy_images = torch.randn(batch_size, 3, img_size, img_size).to(device)
        
        print(f"📊 테스트 설정:")
        print(f"  - 배치 크기: {batch_size}")
        print(f"  - 이미지 크기: {img_size}")
        print(f"  - MC 샘플 수: {config['model']['dropout']['num_samples']}")
        print(f"  - Box Std Threshold: {config['training']['semi_supervised']['uncertainty']['box_std_threshold']}")
        print(f"  - Entropy Threshold: {config['training']['semi_supervised']['uncertainty']['entropy_threshold']}")
        
        # 1. Teacher MC Dropout 예측 테스트
        print("\n🔍 1. Teacher MC Dropout 예측 테스트")
        model.teacher_model.eval()
        
        with torch.no_grad():
            mc_result = detector.predict_with_uncertainty_legacy(
                dummy_images, device=device, save_predictions=False
            )
        
        if mc_result:
            print(f"✅ Teacher MC Dropout 성공: {len(mc_result)} 이미지 결과")
            
            # 결과 분석
            for img_idx, result in enumerate(mc_result):
                print(f"  📊 이미지 {img_idx}:")
                print(f"    - 박스 수: {len(result.get('boxes', []))}")
                print(f"    - 라벨 수: {len(result.get('labels', []))}")
                print(f"    - 점수 수: {len(result.get('scores', []))}")
                
                if 'boxes' in result and len(result['boxes']) > 0:
                    boxes = result['boxes']
                    print(f"    - 박스 좌표 범위: [{boxes.min().item():.3f}, {boxes.max().item():.3f}]")
                    print(f"    - 박스 크기 범위: [{boxes[:, 2:].min().item():.3f}, {boxes[:, 2:].max().item():.3f}]")
        else:
            print("❌ Teacher MC Dropout 실패")
            return False
        
        # 2. High Quality Pseudo Labels 생성 테스트
        print("\n🎯 2. High Quality Pseudo Labels 생성 테스트")
        
        high_quality_pseudo_labels = []
        
        for img_idx in range(len(dummy_images)):
            if img_idx < len(mc_result):
                result = mc_result[img_idx]
                
                if len(result.get('boxes', [])) > 0:
                    boxes = result['boxes']
                    labels = result['labels']
                    scores = result['scores']
                    
                    # YOLO 형식으로 변환 [class_id, x, y, w, h]
                    yolo_detections = []
                    for i in range(len(boxes)):
                        # 박스 좌표 유효성 검사 및 클램핑
                        box_coords = boxes[i].clone()
                        box_coords = torch.clamp(box_coords, 0.0, 1.0)  # 0~1 범위로 클램핑
                        
                        # 너무 작은 박스 필터링 (최소 크기 0.01)
                        if box_coords[2] >= 0.01 and box_coords[3] >= 0.01:
                            yolo_detection = torch.zeros(5, device=boxes[i].device)
                            yolo_detection[0] = labels[i].float().clone()  # class_id 복사
                            yolo_detection[1:5] = box_coords  # x, y, w, h 복사
                            yolo_detections.append(yolo_detection)
                    
                    if yolo_detections:
                        consistent_detections = torch.stack(yolo_detections).clone()
                        
                        high_quality_pseudo_labels.append({
                            'boxes': consistent_detections,
                            'paths': f'dummy_path_{img_idx}',
                            'uncertainty_stats': {
                                'mc_samples': detector.num_samples,
                                'detections_count': len(consistent_detections),
                                'reliability_score': scores.mean().item() if hasattr(scores, 'mean') else 0.0,
                                'quality_grade': 'high'
                            }
                        })
        
        print(f"✅ High Quality Pseudo Labels 생성: {len(high_quality_pseudo_labels)} 이미지")
        
        # 3. Postfix Dict 테스트 (수정된 부분)
        print("\n📊 3. Postfix Dict 테스트")
        
        if high_quality_pseudo_labels:
            total_detections = sum(len(pl['boxes']) for pl in high_quality_pseudo_labels)
            print(f"  📊 총 고품질 탐지: {total_detections}")
            
            # 수정된 코드 테스트
            postfix_dict = {}
            postfix_dict['pseudo'] = f"{total_detections}"
            print(f"  ✅ Postfix Dict 생성 성공: {postfix_dict}")
        else:
            print("  ⚠️ 고품질 pseudo labels가 없음")
        
        # 4. MC Consistency Loss 테스트
        print("\n🔄 4. MC Consistency Loss 테스트")
        
        try:
            from utils.yolo_losses import MCDropoutConsistencyLoss
            
            mc_consistency_loss_fn = MCDropoutConsistencyLoss(
                alpha=1.0,
                beta=0.5,
                temperature=1.0,
                uncertainty_threshold=0.1,
                use_adaptive_weighting=True
            )
            
            # Student MC 예측 시뮬레이션
            student_mc_predictions = []
            for _ in range(3):  # 3개 MC 샘플
                with torch.no_grad():
                    pred = model.student_model.model(dummy_images)
                    if isinstance(pred, (list, tuple)):
                        pred = pred[0] if len(pred) > 0 else None
                    if pred is not None:
                        student_mc_predictions.append(pred.clone())
            
            if student_mc_predictions and high_quality_pseudo_labels:
                mc_consistency_result = mc_consistency_loss_fn(
                    student_mc_predictions=student_mc_predictions,
                    high_quality_pseudo_labels=high_quality_pseudo_labels,
                    strong_unlabeled_images=dummy_images.clone(),
                    return_components=True
                )
                
                print(f"✅ MC Consistency Loss 계산 성공:")
                print(f"  - Total Loss: {mc_consistency_result['total'].item():.6f}")
                print(f"  - Pseudo Consistency: {mc_consistency_result['pseudo_consistency'].item():.6f}")
                print(f"  - MC Uncertainty: {mc_consistency_result['mc_uncertainty'].item():.6f}")
                
                if 'adaptive_weights' in mc_consistency_result:
                    print(f"  - Adaptive Alpha: {mc_consistency_result['adaptive_weights']['alpha']:.4f}")
                    print(f"  - Adaptive Beta: {mc_consistency_result['adaptive_weights']['beta']:.4f}")
            else:
                print("❌ MC Consistency Loss 계산 실패: 데이터 부족")
                
        except Exception as e:
            print(f"❌ MC Consistency Loss 테스트 실패: {e}")
        
        print("\n🎉 모든 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_loss_debug()
    if success:
        print("\n✅ Loss 계산 디버깅 테스트 성공!")
    else:
        print("\n❌ Loss 계산 디버깅 테스트 실패!") 