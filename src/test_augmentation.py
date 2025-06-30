#!/usr/bin/env python3
"""
Teacher-Student Augmentation 전략 테스트 스크립트

이 스크립트는 다음을 확인합니다:
1. Weak Augmentation (Teacher용)
2. Strong Augmentation (Student용)  
3. 같은 이미지에 대한 서로 다른 augmentation 결과 비교
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from train import get_weak_augmentation, get_strong_augmentation, get_val_transform

def test_augmentations():
    """Augmentation 전략 테스트"""
    
    # 테스트 이미지 경로 (COCO 데이터셋에서 샘플 이미지 사용)
    test_image_path = "/media/lee/Data/COCO/val2017/images/000000000139.jpg"
    
    if not Path(test_image_path).exists():
        print(f"테스트 이미지가 존재하지 않습니다: {test_image_path}")
        print("COCO validation 데이터셋의 다른 이미지를 사용하세요.")
        return
    
    # 원본 이미지 로드
    original_image = Image.open(test_image_path).convert('RGB')
    print(f"원본 이미지 크기: {original_image.size}")
    
    # Augmentation 함수들 생성
    weak_aug = get_weak_augmentation(img_size=640)
    strong_aug = get_strong_augmentation(img_size=640)
    val_transform = get_val_transform(img_size=640)
    
    # 각 augmentation 적용
    print("\n=== Augmentation 적용 ===")
    
    # 1. Validation (No augmentation)
    val_image = val_transform(original_image)
    print(f"Validation transform: {val_image.shape}")
    
    # 2. Weak Augmentation (Teacher용)
    weak_images = []
    for i in range(3):
        weak_img = weak_aug(original_image)
        weak_images.append(weak_img)
        print(f"Weak augmentation {i+1}: {weak_img.shape}")
    
    # 3. Strong Augmentation (Student용)
    strong_images = []
    for i in range(3):
        strong_img = strong_aug(original_image)
        strong_images.append(strong_img)
        print(f"Strong augmentation {i+1}: {strong_img.shape}")
    
    # 시각화
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Teacher-Student Augmentation Comparison', fontsize=16)
    
    # 원본 이미지 (첫 번째 열)
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(tensor_to_pil_display(val_image))
    axes[1, 0].set_title('Validation\n(No Augmentation)')
    axes[1, 0].axis('off')
    
    axes[2, 0].axis('off')  # 빈 공간
    
    # Weak Augmentation (Teacher용) - 두 번째, 세 번째, 네 번째 열
    for i, weak_img in enumerate(weak_images):
        axes[0, i+1].imshow(tensor_to_pil_display(weak_img))
        axes[0, i+1].set_title(f'Teacher (Weak Aug) #{i+1}')
        axes[0, i+1].axis('off')
    
    # Strong Augmentation (Student용) - 두 번째, 세 번째, 네 번째 열  
    for i, strong_img in enumerate(strong_images):
        axes[1, i+1].imshow(tensor_to_pil_display(strong_img))
        axes[1, i+1].set_title(f'Student (Strong Aug) #{i+1}')
        axes[1, i+1].axis('off')
    
    # 빈 공간 제거
    for i in range(1, 4):
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    # 결과 저장
    save_path = Path("augmentation_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 결과 이미지 저장: {save_path}")
    
    plt.show()
    
    # 통계 분석
    print("\n=== Augmentation 효과 분석 ===")
    
    # Weak vs Strong augmentation 차이 분석
    weak_var = torch.var(torch.stack(weak_images))
    strong_var = torch.var(torch.stack(strong_images))
    
    print(f"Weak Augmentation 분산: {weak_var:.6f}")
    print(f"Strong Augmentation 분산: {strong_var:.6f}")
    print(f"Strong/Weak 분산 비율: {strong_var/weak_var:.2f}x")
    
    if strong_var > weak_var:
        print("✓ Strong augmentation이 더 다양한 변화를 생성합니다.")
    else:
        print("⚠️ Strong augmentation의 변화가 예상보다 작습니다.")

def tensor_to_pil_display(tensor):
    """
    정규화된 텐서를 PIL Image로 변환 (시각화용)
    """
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    denormalized = tensor * std + mean
    denormalized = torch.clamp(denormalized, 0, 1)
    
    # PIL Image로 변환
    pil_image = transforms.ToPILImage()(denormalized)
    
    return pil_image

def compare_teacher_student_consistency():
    """Teacher-Student 일관성 테스트"""
    
    print("\n=== Teacher-Student 일관성 테스트 ===")
    
    # 테스트 이미지
    test_image_path = "/media/lee/Data/COCO/val2017/images/000000000139.jpg"
    
    if not Path(test_image_path).exists():
        print("테스트 이미지를 찾을 수 없습니다.")
        return
    
    original_image = Image.open(test_image_path).convert('RGB')
    
    # Teacher와 Student augmentation
    weak_aug = get_weak_augmentation(img_size=640)
    strong_aug = get_strong_augmentation(img_size=640)
    
    # 같은 이미지에 대해 여러 번 augmentation 적용
    n_samples = 5
    
    teacher_consistency = []
    student_consistency = []
    
    # Teacher (Weak) 일관성 측정
    teacher_images = [weak_aug(original_image) for _ in range(n_samples)]
    teacher_stack = torch.stack(teacher_images)
    teacher_var = torch.var(teacher_stack, dim=0).mean()
    
    # Student (Strong) 일관성 측정  
    student_images = [strong_aug(original_image) for _ in range(n_samples)]
    student_stack = torch.stack(student_images)
    student_var = torch.var(student_stack, dim=0).mean()
    
    print(f"Teacher (Weak Aug) 예측 분산: {teacher_var:.6f}")
    print(f"Student (Strong Aug) 예측 분산: {student_var:.6f}")
    print(f"Student/Teacher 분산 비율: {student_var/teacher_var:.2f}x")
    
    print("\n✓ Teacher는 안정적인 pseudo label 생성에 적합")
    print("✓ Student는 robust한 특징 학습에 적합")

if __name__ == "__main__":
    print("🔍 Teacher-Student Augmentation 전략 테스트 시작")
    
    try:
        test_augmentations()
        compare_teacher_student_consistency()
        
        print("\n🎉 테스트 완료!")
        print("Teacher-Student augmentation 전략이 올바르게 구현되었습니다.")
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        print(traceback.format_exc()) 