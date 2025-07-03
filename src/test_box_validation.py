#!/usr/bin/env python3
"""
박스 좌표 유효성 검사 테스트 스크립트
마이너스 값 문제 해결 확인
"""

import torch
import numpy as np

def test_box_validation():
    """박스 좌표 유효성 검사 테스트"""
    print("🧪 박스 좌표 유효성 검사 테스트 시작")
    
    # 테스트 케이스 1: 마이너스 값이 포함된 박스들
    test_boxes = torch.tensor([
        [0.5, 0.3, 0.2, 0.1],    # 정상 박스
        [-0.1, 0.4, 0.3, 0.2],   # x 좌표가 마이너스
        [0.6, -0.2, 0.1, 0.15],  # y 좌표가 마이너스
        [0.7, 0.5, -0.05, 0.2],  # width가 마이너스
        [0.3, 0.8, 0.2, -0.1],   # height가 마이너스
        [1.2, 0.4, 0.3, 0.2],    # x 좌표가 1 초과
        [0.5, 1.1, 0.1, 0.15],   # y 좌표가 1 초과
        [0.2, 0.3, 0.005, 0.1],  # 너무 작은 width
        [0.4, 0.6, 0.2, 0.005],  # 너무 작은 height
    ])
    
    print(f"📊 원본 박스들:")
    print(test_boxes)
    
    # 박스 좌표 유효성 검사 및 클램핑
    valid_boxes = []
    for i, box in enumerate(test_boxes):
        # 박스 좌표를 0~1 범위로 클램핑
        box_coords = torch.clamp(box, 0.0, 1.0)
        
        # 너무 작은 박스 필터링 (최소 크기 0.01)
        if box_coords[2] >= 0.01 and box_coords[3] >= 0.01:
            valid_boxes.append(box_coords)
            print(f"✅ 박스 {i}: 유효함 - {box_coords}")
        else:
            print(f"❌ 박스 {i}: 필터링됨 - 너무 작음 - {box_coords}")
    
    if valid_boxes:
        valid_boxes_tensor = torch.stack(valid_boxes)
        print(f"\n🎯 유효한 박스들 ({len(valid_boxes)}개):")
        print(valid_boxes_tensor)
        
        # 추가 검증
        print(f"\n🔍 검증 결과:")
        print(f"  - 모든 x 좌표 >= 0: {torch.all(valid_boxes_tensor[:, 0] >= 0)}")
        print(f"  - 모든 y 좌표 >= 0: {torch.all(valid_boxes_tensor[:, 1] >= 0)}")
        print(f"  - 모든 width >= 0.01: {torch.all(valid_boxes_tensor[:, 2] >= 0.01)}")
        print(f"  - 모든 height >= 0.01: {torch.all(valid_boxes_tensor[:, 3] >= 0.01)}")
        print(f"  - 모든 좌표 <= 1: {torch.all(valid_boxes_tensor <= 1.0)}")
    else:
        print("❌ 유효한 박스가 없습니다.")
    
    # 테스트 케이스 2: 실제 MC Dropout 결과 시뮬레이션
    print(f"\n🧪 MC Dropout 결과 시뮬레이션")
    
    # 마이너스 값이 포함된 MC 결과 시뮬레이션
    mc_boxes = torch.tensor([
        [34.0000,  4.1569, -0.5109, -0.6950,  0.8612],  # 마이너스 값 포함
        [34.0000,  4.8701, -0.5398, -0.1112,  1.0605],  # 마이너스 값 포함
        [12.0000, 10.8090,  4.5014,  3.2890,  0.7935],  # 정상
        [14.0000,  5.2686,  4.8985,  0.9379, -0.0232],  # 마이너스 값 포함
    ])
    
    print(f"📊 MC Dropout 원본 박스들:")
    print(mc_boxes)
    
    # YOLO 형식으로 변환하면서 유효성 검사
    yolo_detections = []
    for i, box in enumerate(mc_boxes):
        # 박스 좌표 유효성 검사 및 클램핑
        box_coords = box[1:5].clone()  # x, y, w, h만 추출
        box_coords = torch.clamp(box_coords, 0.0, 1.0)  # 0~1 범위로 클램핑
        
        # 너무 작은 박스 필터링 (최소 크기 0.01)
        if box_coords[2] >= 0.01 and box_coords[3] >= 0.01:
            yolo_detection = torch.zeros(5)
            yolo_detection[0] = box[0].float()  # class_id
            yolo_detection[1:5] = box_coords  # x, y, w, h
            yolo_detections.append(yolo_detection)
            print(f"✅ MC 박스 {i}: 유효함 - class_id: {box[0]}, coords: {box_coords}")
        else:
            print(f"❌ MC 박스 {i}: 필터링됨 - 너무 작음 - coords: {box_coords}")
    
    if yolo_detections:
        consistent_detections = torch.stack(yolo_detections)
        print(f"\n🎯 유효한 MC 박스들 ({len(yolo_detections)}개):")
        print(consistent_detections)
        
        # 추가 검증
        print(f"\n🔍 MC 검증 결과:")
        print(f"  - 모든 x 좌표 >= 0: {torch.all(consistent_detections[:, 1] >= 0)}")
        print(f"  - 모든 y 좌표 >= 0: {torch.all(consistent_detections[:, 2] >= 0)}")
        print(f"  - 모든 width >= 0.01: {torch.all(consistent_detections[:, 3] >= 0.01)}")
        print(f"  - 모든 height >= 0.01: {torch.all(consistent_detections[:, 4] >= 0.01)}")
        print(f"  - 모든 좌표 <= 1: {torch.all(consistent_detections[:, 1:5] <= 1.0)}")
    else:
        print("❌ 유효한 MC 박스가 없습니다.")
    
    print(f"\n✅ 박스 좌표 유효성 검사 테스트 완료")

if __name__ == "__main__":
    test_box_validation() 