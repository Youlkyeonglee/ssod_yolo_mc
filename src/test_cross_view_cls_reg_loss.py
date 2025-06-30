#!/usr/bin/env python3
"""
Cross-View Consistency Loss (Classification + Regression) 테스트 스크립트

Teacher-Student MC Dropout 시스템에서 개선된 Cross-View Consistency Loss를 테스트합니다.
- Classification Loss (KL Divergence)
- Regression Loss (Smooth L1)  
- Objectness Loss (Binary Cross Entropy)
"""

import torch
import torch.nn as nn
import sys
import os
import yaml
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uncertainty.mc_dropout import MCDropoutDetector
from models.yolo_mc import YOLOWithMCDropout

def create_mock_predictions(batch_size=1, num_detections=100, num_classes=80):
    """Mock YOLO 예측 생성 (테스트용)"""
    # YOLO 예측 형식: [x, y, w, h, objectness, class1, class2, ...]
    prediction = torch.randn(batch_size, num_detections, 5 + num_classes)
    
    # Box coordinates 정규화 (0-1 범위)
    prediction[..., :4] = torch.sigmoid(prediction[..., :4])
    
    # Objectness score
    prediction[..., 4] = torch.sigmoid(prediction[..., 4])
    
    # Class scores (logits)
    prediction[..., 5:] = torch.randn_like(prediction[..., 5:])
    
    return prediction

def test_cross_view_consistency_loss():
    """Cross-View Consistency Loss 테스트"""
    print("🔬 Cross-View Consistency Loss (Cls + Reg) 테스트 시작")
    print("=" * 60)
    
    # 설정 파일 로드
    config_path = Path("configs/yolo_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 모델 및 MC Dropout 탐지기 생성
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  디바이스: {device}")
    
    # Mock 모델 생성
    model = nn.Sequential(
        nn.Linear(10, 100),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(100, 96)  # COCO 80 classes + 16 (5 + 11 padding)
    )
    
    detector = MCDropoutDetector(
        model=model,
        num_samples=5,
        dropout_rate=0.1,
        box_std_threshold=0.1,
        entropy_threshold=0.5
    )
    
    # Teacher와 Student MC Dropout 예측 생성
    print("\n📊 Mock 예측 데이터 생성")
    num_samples = 5
    batch_size = 2
    num_detections = 50
    num_classes = 80
    
    # Teacher 예측 (Weak Augmentation - 더 안정적)
    teacher_predictions = []
    for i in range(num_samples):
        # Teacher는 더 일관된 예측을 생성 (낮은 분산)
        base_pred = create_mock_predictions(batch_size, num_detections, num_classes)
        noise = torch.randn_like(base_pred) * 0.05  # 낮은 노이즈
        teacher_pred = base_pred + noise
        teacher_predictions.append(teacher_pred)
    
    # Student 예측 (Strong Augmentation - 더 다양함)
    student_predictions = []
    for i in range(num_samples):
        # Student는 더 다양한 예측을 생성 (높은 분산)
        base_pred = create_mock_predictions(batch_size, num_detections, num_classes)
        noise = torch.randn_like(base_pred) * 0.15  # 높은 노이즈
        student_pred = base_pred + noise
        student_predictions.append(student_pred)
    
    print(f"✅ Teacher 예측: {len(teacher_predictions)}개 샘플")
    print(f"✅ Student 예측: {len(student_predictions)}개 샘플")
    print(f"📏 예측 형태: {teacher_predictions[0].shape}")
    
    # Cross-View Consistency Loss 계산 테스트
    print("\n🧮 Cross-View Consistency Loss 계산")
    
    # 1. 기본 설정으로 테스트
    print("\n1️⃣ 기본 설정 테스트")
    cross_view_loss = detector.calculate_cross_view_consistency_loss(
        teacher_predictions=teacher_predictions,
        student_predictions=student_predictions,
        weight=0.1,
        cls_weight=1.0,
        reg_weight=2.0,
        obj_weight=1.0
    )
    
    print(f"📈 Cross-View Consistency Loss: {cross_view_loss.item():.6f}")
    
    # Loss 세부 정보 출력
    if hasattr(cross_view_loss, 'loss_info'):
        loss_info = cross_view_loss.loss_info
        print(f"  📦 Regression Loss: {loss_info['reg_loss'].item():.6f}")
        print(f"  🎯 Objectness Loss: {loss_info['obj_loss'].item():.6f}")
        print(f"  🏷️  Classification Loss: {loss_info['cls_loss'].item():.6f}")
        print(f"  ⚖️  가중치 - Reg: {loss_info['weights']['reg_weight']}, Obj: {loss_info['weights']['obj_weight']}, Cls: {loss_info['weights']['cls_weight']}")
    
    # 2. 다양한 가중치로 테스트
    print("\n2️⃣ 다양한 가중치 테스트")
    test_configs = [
        {"cls_weight": 2.0, "reg_weight": 1.0, "obj_weight": 1.0, "name": "Classification 중심"},
        {"cls_weight": 1.0, "reg_weight": 3.0, "obj_weight": 1.0, "name": "Regression 중심"},
        {"cls_weight": 1.0, "reg_weight": 1.0, "obj_weight": 2.0, "name": "Objectness 중심"},
        {"cls_weight": 0.0, "reg_weight": 1.0, "obj_weight": 1.0, "name": "Classification 비활성화"}
    ]
    
    for test_config in test_configs:
        loss = detector.calculate_cross_view_consistency_loss(
            teacher_predictions=teacher_predictions,
            student_predictions=student_predictions,
            weight=0.1,
            cls_weight=test_config["cls_weight"],
            reg_weight=test_config["reg_weight"],
            obj_weight=test_config["obj_weight"]
        )
        
        print(f"  {test_config['name']}: {loss.item():.6f}")
        if hasattr(loss, 'loss_info'):
            info = loss.loss_info
            print(f"    Reg: {info['reg_loss'].item():.4f}, Obj: {info['obj_loss'].item():.4f}, Cls: {info['cls_loss'].item():.4f}")
    
    # 3. 빈 예측 처리 테스트
    print("\n3️⃣ 예외 상황 테스트")
    
    # 빈 Teacher 예측
    empty_loss = detector.calculate_cross_view_consistency_loss(
        teacher_predictions=[],
        student_predictions=student_predictions,
        weight=0.1
    )
    print(f"  빈 Teacher 예측: {empty_loss.item():.6f}")
    
    # 빈 Student 예측
    empty_loss = detector.calculate_cross_view_consistency_loss(
        teacher_predictions=teacher_predictions,
        student_predictions=[],
        weight=0.1
    )
    print(f"  빈 Student 예측: {empty_loss.item():.6f}")
    
    # 4. 크기 불일치 테스트
    print("\n4️⃣ 크기 불일치 처리 테스트")
    
    # Teacher는 더 작은 feature map
    small_teacher_predictions = []
    for pred in teacher_predictions:
        small_pred = pred[..., :50]  # 더 작은 feature 크기
        small_teacher_predictions.append(small_pred)
    
    size_mismatch_loss = detector.calculate_cross_view_consistency_loss(
        teacher_predictions=small_teacher_predictions,
        student_predictions=student_predictions,
        weight=0.1
    )
    print(f"  크기 불일치 처리: {size_mismatch_loss.item():.6f}")
    
    # 5. 설정 파일 기반 테스트
    print("\n5️⃣ 설정 파일 기반 테스트")
    cross_view_config = config['training']['loss']['semi_supervised'].get('cross_view_consistency', {})
    
    if cross_view_config.get('enabled', True):
        config_loss = detector.calculate_cross_view_consistency_loss(
            teacher_predictions=teacher_predictions,
            student_predictions=student_predictions,
            weight=cross_view_config.get('weight', 0.1),
            cls_weight=cross_view_config.get('cls_weight', 1.0),
            reg_weight=cross_view_config.get('reg_weight', 2.0),
            obj_weight=cross_view_config.get('obj_weight', 1.0)
        )
        print(f"  설정 파일 기반: {config_loss.item():.6f}")
        print(f"  설정값 - Weight: {cross_view_config.get('weight', 0.1)}")
        print(f"         - Cls: {cross_view_config.get('cls_weight', 1.0)}, Reg: {cross_view_config.get('reg_weight', 2.0)}, Obj: {cross_view_config.get('obj_weight', 1.0)}")
    
    print("\n" + "=" * 60)
    print("✅ Cross-View Consistency Loss (Cls + Reg) 테스트 완료!")
    print("\n📋 테스트 결과 요약:")
    print("  - Classification Loss: KL Divergence로 확률 분포 일치")
    print("  - Regression Loss: Smooth L1으로 Bounding Box 일치")
    print("  - Objectness Loss: Binary Cross Entropy로 객체 신뢰도 일치")
    print("  - 가중치 조절: 각 Loss 컴포넌트별 독립적 조절 가능")
    print("  - 예외 처리: 빈 예측, 크기 불일치 상황 안전 처리")
    print("  - 설정 파일 연동: YAML 설정으로 하이퍼파라미터 관리")

if __name__ == "__main__":
    test_cross_view_consistency_loss() 