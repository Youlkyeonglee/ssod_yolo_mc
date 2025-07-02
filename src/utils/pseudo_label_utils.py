"""
Pseudo Label 생성 및 관리 유틸리티
Teacher MC Dropout을 활용한 고품질 pseudo label 생성
"""

import torch
from typing import List, Dict, Any
from pathlib import Path


def get_weak_augmentation(img_size: int = 640):
    """Weak Augmentation for Teacher model (안정적인 pseudo label 생성)"""
    import torchvision.transforms as transforms
    
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    정규화된 텐서를 원본 이미지로 복원
    
    Args:
        tensor: 정규화된 이미지 텐서 (C, H, W)
        mean: 정규화에 사용된 평균값
        std: 정규화에 사용된 표준편차
    
    Returns:
        denormalized tensor
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).view(-1, 1, 1)
    if isinstance(std, list):
        std = torch.tensor(std).view(-1, 1, 1)
    
    # GPU 텐서인 경우 CPU로 이동
    if tensor.is_cuda:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    
    # Denormalize: x = x * std + mean
    denormalized = tensor * std + mean
    
    # 0-1 범위로 클램핑
    denormalized = torch.clamp(denormalized, 0, 1)
    
    return denormalized


def tensor_to_pil(tensor):
    """
    정규화된 텐서를 PIL Image로 변환
    
    Args:
        tensor: 정규화된 이미지 텐서 (C, H, W)
    
    Returns:
        PIL Image
    """
    import torchvision.transforms as transforms
    
    # Denormalize
    denormalized = denormalize_tensor(tensor)
    
    # PIL Image로 변환
    pil_image = transforms.ToPILImage()(denormalized.cpu())
    
    return pil_image


def update_pseudo_labels(model, unlabeled_loader, detector, conf_threshold, device, config):
    """
    Teacher MC Dropout으로 Weak Augmentation된 unlabeled 데이터에서 고품질 pseudo label 생성
    
    연구 시나리오:
    - Teacher 모델: Weak Augmentation으로 안정적인 pseudo label 생성
    - MC Dropout 불확실성 추정으로 고품질 pseudo label 필터링
    - 예측 분산이 낮을수록 신뢰도 높은 detection으로 판정
    
    Args:
        model: Teacher-Student YOLO 모델
        unlabeled_loader: Unlabeled 데이터 로더
        detector: MC Dropout 탐지기
        conf_threshold: 신뢰도 임계값
        device: 실행 디바이스
        config: 설정 딕셔너리
    
    Returns:
        List[Dict]: 고품질 pseudo label 리스트
    """
    # DDP 지원
    model_for_pseudo = model.module if hasattr(model, 'module') else model
    model_for_pseudo.eval()  # Teacher는 eval 모드에서 MC Dropout
    pseudo_labels = []
    
    # Teacher용 Weak Augmentation
    weak_transform = get_weak_augmentation(config['data']['img_size'])
    
    # MC Dropout 설정
    num_mc_samples = config['model']['dropout']['num_samples']
    
    print(f"🔍 Teacher MC Dropout Pseudo Label 생성 시작 (MC samples: {num_mc_samples})")
    
    import time
    start_time = time.time()
    timeout_seconds = 60  # 60초 타임아웃
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(unlabeled_loader):
            # 타임아웃 체크
            if time.time() - start_time > timeout_seconds:
                print(f"⚠️  Pseudo label generation timeout after {timeout_seconds}s")
                break
                
            if batch_idx >= 10:  # 테스트용으로 제한
                break
                
            # === Teacher MC Dropout: Weak Augmentation으로 안정적인 예측 ===
            weak_images = []
            for img in batch['images']:
                if isinstance(img, torch.Tensor):
                    img_pil = tensor_to_pil(img)
                else:
                    img_pil = img
                
                # Teacher용 Weak augmentation 적용
                weak_img = weak_transform(img_pil)
                weak_images.append(weak_img)
            
            weak_images = torch.stack(weak_images).to(device)
            
            # Teacher MC Dropout 단일 호출 (내부적으로 num_samples만큼 MC 샘플링 수행)
            try:
                model_for_pseudo.eval()  # Teacher는 eval 모드에서 MC Dropout
                with torch.no_grad():
                    # === 개선된 단일 MC Dropout 호출 (다층 신뢰도 평가 시스템) ===
                    # predict_with_uncertainty 함수가 다층 신뢰도 평가로 고품질 pseudo label 생성
                    # DDP 지원: 디바이스 정보 명확하게 전달
                    actual_device = weak_images.device
                    print(f"  📊 Processing batch {batch_idx+1}, device: {actual_device}")
                    
                    mc_result = detector.predict_with_uncertainty(
                        weak_images, 
                        device=str(actual_device), 
                        config=config
                    )
                
                if mc_result:
                    # 기존 코드와의 호환성을 위해 리스트로 래핑
                    mc_results_list = [mc_result]
                    
                    print(f"  📊 MC Dropout completed: {len(mc_result)} image results with {detector.num_samples} internal samples each")
                    
                    if len(mc_results_list) >= 1:  # 최소 1개 결과 필요
                        # === MC Dropout 불확실성 기반 고품질 Pseudo Label 선별 ===
                        
                        # 각 detection에 대해 MC 샘플들의 분산 계산
                        high_quality_pseudo_labels = []
                        
                        for img_idx in range(len(weak_images)):
                            # 해당 이미지의 모든 MC 샘플 수집
                            img_mc_detections = []
                            
                            for mc_results in mc_results_list:
                                if img_idx < len(mc_results):
                                    result = mc_results[img_idx]
                                    if len(result.get('boxes', [])) > 0:
                                        img_mc_detections.append(result)
                                    
                                    # MC Dropout predict_with_uncertainty에서 이미 고품질 pseudo label을 생성했으므로
                                    # 중복 필터링 없이 직접 사용
                                    
                                    if len(img_mc_detections) >= 1:
                                        # predict_with_uncertainty에서 이미 다층 신뢰도 평가를 통해 필터링된 결과 사용
                                        result = img_mc_detections[0]  # 첫 번째 (유일한) MC 결과 사용
                                        
                                        # 이미 필터링된 고품질 detection이 있는지 확인
                                        if len(result.get('boxes', [])) > 0:
                                            # YOLO 형식으로 변환 [class_id, x, y, w, h]
                                            boxes = result['boxes']
                                            labels = result['labels']
                                            
                                            yolo_detections = []
                                            for i in range(len(boxes)):
                                                yolo_detection = torch.zeros(5)
                                                yolo_detection[0] = labels[i].float()  # class_id
                                                yolo_detection[1:5] = boxes[i]  # x, y, w, h
                                                yolo_detections.append(yolo_detection)
                                        
                                            if yolo_detections:
                                                consistent_detections = torch.stack(yolo_detections)
                                                
                                            high_quality_pseudo_labels.append({
                                                'boxes': consistent_detections,
                                                'image_path': batch.get('paths', [''])[img_idx] if 'paths' in batch and img_idx < len(batch.get('paths', [])) else '',
                                                'uncertainty_stats': {
                                                    'mc_samples': detector.num_samples,  # 내부 MC 샘플 수
                                                        'detections_count': len(consistent_detections),
                                                        'reliability_score': result.get('reliability_score', 0.0),  # V5 신뢰도 점수
                                                        'quality_grade': result.get('quality_grade', 'Unknown')  # V5 품질 등급
                                                }
                                            })
                        
                        pseudo_labels.extend(high_quality_pseudo_labels)
                        
                        if high_quality_pseudo_labels:
                            total_detections = sum(len(pl['boxes']) for pl in high_quality_pseudo_labels)
                            print(f"  📊 Batch {batch_idx}: {len(high_quality_pseudo_labels)} images, {total_detections} high-quality detections")
                    
            except Exception as e:
                print(f"❌ Error in Teacher MC Dropout pseudo label generation (batch {batch_idx}): {e}")
                continue
    
    total_time = time.time() - start_time
    print(f"✅ Teacher MC Dropout Pseudo Label 생성 완료: {len(pseudo_labels)} 고품질 pseudo labels in {total_time:.2f}s")
    return pseudo_labels 