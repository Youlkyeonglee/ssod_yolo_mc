#!/usr/bin/env python3
"""
파라미터 크기 불일치 디버깅 스크립트

Teacher-Student 모델 간의 파라미터 차이를 상세히 분석합니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from models.yolo_mc import YOLOWithMCDropout

def debug_parameter_mismatch():
    """파라미터 크기 불일치 상세 분석"""
    print("🔍 Debugging Parameter Mismatch")
    print("=" * 60)
    
    # 모델 생성
    print("\n📦 Creating YOLOWithMCDropout model...")
    model = YOLOWithMCDropout(
        model_name="yolov8m",  # 원래 사용하던 모델
        dropout_rate=0.1,
        num_classes=91,
        ema_decay=0.999,
        feature_alignment_enabled=True
    )
    
    # 상세한 파라미터 분석
    print("\n🔬 Analyzing parameter structure...")
    
    # Named parameters로 더 상세한 정보 확인
    teacher_named_params = dict(model.teacher_model.model.named_parameters())
    student_named_params = dict(model.student_model.model.named_parameters())
    
    print(f"Teacher named parameters: {len(teacher_named_params)}")
    print(f"Student named parameters: {len(student_named_params)}")
    
    # 파라미터 이름 비교
    teacher_names = set(teacher_named_params.keys())
    student_names = set(student_named_params.keys())
    
    only_teacher = teacher_names - student_names
    only_student = student_names - teacher_names
    common_names = teacher_names & student_names
    
    print(f"\nParameters only in teacher: {len(only_teacher)}")
    for name in list(only_teacher)[:5]:
        print(f"  - {name}: {teacher_named_params[name].shape}")
    
    print(f"\nParameters only in student: {len(only_student)}")
    for name in list(only_student)[:5]:
        print(f"  - {name}: {student_named_params[name].shape}")
    
    print(f"\nCommon parameters: {len(common_names)}")
    
    # 공통 파라미터 중 크기가 다른 것들 찾기
    shape_mismatches = []
    for name in common_names:
        t_shape = teacher_named_params[name].shape
        s_shape = student_named_params[name].shape
        if t_shape != s_shape:
            shape_mismatches.append((name, t_shape, s_shape))
    
    print(f"\n❌ Shape mismatches found: {len(shape_mismatches)}")
    for name, t_shape, s_shape in shape_mismatches[:10]:  # 처음 10개만 표시
        print(f"  - {name}:")
        print(f"    Teacher: {t_shape}")
        print(f"    Student: {s_shape}")
    
    if len(shape_mismatches) > 10:
        print(f"    ... and {len(shape_mismatches) - 10} more mismatches")
    
    # 모델 구조 자체 비교
    print(f"\n🏗️  Model structure analysis...")
    
    # Teacher 모델 구조
    print(f"Teacher model type: {type(model.teacher_model.model)}")
    print(f"Student model type: {type(model.student_model.model)}")
    
    # 클래스 수 확인
    teacher_nc = getattr(model.teacher_model.model, 'nc', 'Not found')
    student_nc = getattr(model.student_model.model, 'nc', 'Not found')
    print(f"Teacher nc: {teacher_nc}")
    print(f"Student nc: {student_nc}")
    
    # State dict 크기 비교
    teacher_state = model.teacher_model.model.state_dict()
    student_state = model.student_model.model.state_dict()
    
    print(f"\nState dict keys:")
    print(f"Teacher: {len(teacher_state)} keys")
    print(f"Student: {len(student_state)} keys")
    
    # 불일치 원인 분석
    print(f"\n💡 Potential causes of mismatch:")
    
    # 1. MC Dropout hooks 영향 확인
    student_hooks = 0
    teacher_hooks = 0
    for module in model.student_model.model.modules():
        if hasattr(module, '_mc_dropout_hooks'):
            student_hooks += len(module._mc_dropout_hooks)
    
    for module in model.teacher_model.model.modules():
        if hasattr(module, '_mc_dropout_hooks'):
            teacher_hooks += len(module._mc_dropout_hooks)
    
    print(f"1. MC Dropout hooks - Teacher: {teacher_hooks}, Student: {student_hooks}")
    
    # 2. 클래스 수 설정 확인
    print(f"2. Class count settings:")
    print(f"   Expected: 91")
    print(f"   Teacher nc: {teacher_nc}")
    print(f"   Student nc: {student_nc}")
    
    # 3. 모델 초기화 상태 확인
    print(f"3. Model initialization state:")
    print(f"   Teacher training mode: {model.teacher_model.model.training}")
    print(f"   Student training mode: {model.student_model.model.training}")
    
    return shape_mismatches

def fix_parameter_mismatch():
    """파라미터 불일치 수정 시도"""
    print(f"\n🔧 Attempting to fix parameter mismatch...")
    
    # 대안 방법: Teacher를 처음부터 Student와 완전히 동일하게 만들기
    print("Creating models with alternative approach...")
    
    # 1. Student 모델 생성 및 설정
    student_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # 2. Teacher 모델을 Student의 정확한 복사본으로 생성
    import copy
    teacher_model = copy.deepcopy(student_model)
    
    # 3. 파라미터 비교
    teacher_params = list(teacher_model.parameters())
    student_params = list(student_model.parameters())
    
    print(f"Alternative approach:")
    print(f"Teacher params: {len(teacher_params)}")
    print(f"Student params: {len(student_params)}")
    
    mismatches = 0
    for i, (tp, sp) in enumerate(zip(teacher_params, student_params)):
        if tp.shape != sp.shape:
            print(f"Mismatch {i}: {tp.shape} vs {sp.shape}")
            mismatches += 1
    
    print(f"Mismatches: {mismatches}")
    return mismatches == 0

if __name__ == "__main__":
    # 현재 문제 분석
    mismatches = debug_parameter_mismatch()
    
    # 대안 방법 테스트
    if mismatches:
        print(f"\n" + "="*60)
        fix_success = fix_parameter_mismatch()
        if fix_success:
            print("✅ Alternative approach works!")
        else:
            print("❌ Alternative approach also has issues") 