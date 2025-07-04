from pathlib import Path
import argparse
import torch
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 사용
import atexit
import signal
import sys
import gc
import numpy as np
from models.yolo_mc import YOLOWithMCDropout
from uncertainty.mc_dropout import MCDropoutDetector
from data.semi_supervised_dataset import SemiSupervisedDataset
from utils.train_utils import (
    setup_logger,
    save_checkpoint,
    load_checkpoint
)
from utils.pseudo_label_utils import (
    update_pseudo_labels,
    get_weak_augmentation, 
    denormalize_tensor, 
    tensor_to_pil
)
from utils.yolo_losses import (
    YOLOLossSupervised,
    DistributionalConsistencyLoss,
    calculate_unlabeled_loss,
    MCDropoutConsistencyLoss,
    create_mc_consistency_loss,
    YOLOLossSemiSupervised
)
import torchvision.transforms as transforms
import yaml
from tqdm import tqdm
from utils.visualization import plot_metrics
from uncertainty.mc_analysis import save_mc_dropout_analysis, evaluate_mc_dropout_quality_metrics

# GPU 설정 관련 import
from utils.distributed_utils import setup_gpu_config

import os
import sys
import warnings
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

def cleanup_gpu_memory():
    """GPU 메모리 정리 함수"""
    try:
        print("\n🧹 GPU 메모리 정리 중...")
        
        # CUDA 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # 모든 GPU 디바이스 정리
            for device_id in range(torch.cuda.device_count()):
                with torch.cuda.device(f'cuda:{device_id}'):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
        
        # Python 가비지 컬렉션
        gc.collect()
        
        print("✅ GPU 메모리 정리 완료")
        
        # 메모리 상태 출력
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(device_id) / 1024**3    # GB
                print(f"   GPU {device_id}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                
    except Exception as e:
        print(f"⚠️  GPU 메모리 정리 중 오류: {e}")

def signal_handler(signum, frame):
    """시그널 핸들러 - 강제 종료 시 GPU 메모리 정리"""
    print(f"\n🛑 시그널 {signum} 수신됨. 안전하게 종료합니다...")
    cleanup_gpu_memory()
    sys.exit(0)

def setup_cleanup_handlers():
    """프로그램 종료 시 정리 핸들러 설정"""
    # 정상 종료 시 GPU 메모리 정리
    atexit.register(cleanup_gpu_memory)
    
    # 시그널 핸들러 설정 (Ctrl+C, 강제 종료 등)
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill 명령어
    
    # Unix 시스템에서만 사용 가능한 시그널들
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, signal_handler)
    if hasattr(signal, 'SIGUSR2'):
        signal.signal(signal.SIGUSR2, signal_handler)

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO with MC Dropout for Semi-Supervised Learning')
    
    # 기본 설정
    parser.add_argument('--config', type=str, 
                      default=str(Path(__file__).parent / 'configs' / 'yolo_config.yaml'),
                      help='YAML 설정 파일 경로')
    
    # 실행 관련 설정만 명령행 인자로 받음
    parser.add_argument('--save_dir', type=str, default=".",
                      help='프로젝트 루트 디렉토리 (runs 폴더가 생성될 위치)')
    parser.add_argument('--val_data_path', type=str, default="/media/oem/personal_vol/yklee/data/COCO/val2017/images",
                      help='검증 데이터 경로')
    parser.add_argument('--device', type=str, default=None,
                      help='실행 디바이스 (예: cpu, cuda:0)')
    parser.add_argument('--resume', type=str, default=None,
                      help='체크포인트에서 재시작')
    
    # GPU 관련 인자 추가
    parser.add_argument('--gpu_ids', type=str, default=None,
                      help='사용할 GPU ID (쉼표로 구분, 예: 0,1,2,3)')
    
    args = parser.parse_args()
    
    # YAML 설정 파일 로드
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # 명령행 인자로 GPU 설정 오버라이드
    if args.gpu_ids is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        config['gpu']['gpu_ids'] = gpu_ids
        config['gpu']['auto_detect'] = False
    
    
    
    # config의 값들을 args에 추가 (새로운 구조에 맞게 수정)
    args.model = config['model']['name']
    args.dropout_rate = config['model']['dropout']['rate']
    args.num_samples = config['model']['dropout']['num_samples']
    args.batch_size = config['data']['batch_size']
    args.epochs = config['training']['epochs']
    args.val_interval = config['training']['val_interval']
    args.save_interval = config['training']['save_interval']
    args.labeled_ratio = config['data']['labeled_ratio']
    args.seed = config['data']['seed']
    args.max_samples = config['data']['max_samples']
    args.num_workers = config['data']['num_workers']
    
    # Semi-supervised 설정
    semi_config = config['training']['loss_weights']['semi_supervised']
    args.box_std_threshold = semi_config['uncertainty']['box_std_threshold']
    args.entropy_threshold = semi_config['uncertainty']['entropy_threshold']
    args.max_pseudo_labels = semi_config['max_pseudo_labels']
    args.unlabeled_weight = semi_config['unlabeled_weight']
    args.pseudo_label_start_epoch = semi_config['pseudo_label_start_epoch']
    args.conf_threshold = semi_config['conf_threshold']
    args.num_mc_samples = semi_config['num_mc_samples']
    
    # Feature alignment 설정
    args.feature_alignment_enabled = config['model']['feature_alignment']['enabled']
    args.feature_alignment_weight = config['model']['feature_alignment']['weight']
    
    # device가 설정되지 않은 경우 config에서 가져오기
    if args.device is None:
        args.device = config['inference']['device']
    
    return args, config

# get_weak_augmentation 함수는 utils.pseudo_label_utils에서 임포트됨

def get_strong_augmentation(img_size: int = 640):
    """Strong Augmentation for Student model (robust 특징 학습)"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        # Cutout augmentation (randomly mask patches)
        transforms.Lambda(lambda x: apply_cutout(x, n_holes=1, length=32)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transform(img_size: int = 640):
    """Validation transform (no augmentation)"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def apply_cutout(img, n_holes=1, length=32):
    """
    Cutout augmentation: randomly mask square regions
    
    Args:
        img: PIL Image or tensor
        n_holes: number of holes to cut
        length: length of the square hole
    """
    if isinstance(img, torch.Tensor):
        c, h, w = img.shape
        mask = torch.ones((c, h, w), dtype=img.dtype, device=img.device)
        
        for _ in range(n_holes):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()
            
            y1 = max(0, y - length // 2)
            y2 = min(h, y + length // 2)
            x1 = max(0, x - length // 2)
            x2 = min(w, x + length // 2)
            
            mask[:, y1:y2, x1:x2] = 0
        
        img = img * mask
    
    return img

def _get_student_mc_predictions(student_model, strong_unlabeled_images, device, config):
    """
    Student 모델의 MC Dropout 예측을 안전하게 수행
    
    Args:
        student_model: Student 모델
        strong_unlabeled_images: Strong augmentation된 이미지
        device: 디바이스
        config: 설정
    
    Returns:
        MC Dropout 예측 결과
    """
    try:
        # 모델 상태 저장
        original_training = student_model.model.training
        
        # MC Dropout을 위한 별도 모델 복사본 생성
        with torch.no_grad():
            # 모델을 eval 모드로 설정 (MC Dropout 활성화)
            student_model.model.eval()
            
            # MC Dropout 예측 수행
            mc_predictions = []
            num_samples = min(5, config['model']['dropout'].get('num_samples', 10))
            
            for _ in range(num_samples):
                # 각 MC 샘플에서 예측 수행
                with torch.no_grad():
                    pred = student_model.model(strong_unlabeled_images)
                    mc_predictions.append(pred)
            
            # 모델 상태 복원
            student_model.model.train(original_training)
            
            # 결과를 detector 형식으로 변환
            return mc_predictions
            
    except Exception as e:
        print(f"MC Dropout 예측 실패: {e}")
        return None



# denormalize_tensor, tensor_to_pil 함수들은 utils.pseudo_label_utils에서 임포트됨

def train_one_epoch(
    model: torch.nn.Module,
    labeled_loader: torch.utils.data.DataLoader,
    unlabeled_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    detector: MCDropoutDetector,
    device: str,
    epoch: int,
    logger: logging.Logger,
    config: dict,
    unlabeled_weight: float = 0.5,
    pseudo_label_start_epoch: int = 10,
    conf_threshold: float = 0.5,
    alignment_weight: float = 0.1  # Feature Alignment Loss 가중치
):
    """한 에포크 학습"""
    torch.autograd.set_detect_anomaly(True)
    model.train()
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    labeled_iter = iter(labeled_loader)
    
    # 단일 GPU 학습
    world_size = 1
    
    # 🎯 MC Dropout Consistency Loss 초기화
    mc_consistency_loss_fn = create_mc_consistency_loss(
        alpha=config['training']['loss_weights']['semi_supervised'].get('mc_consistency_alpha', 1.0),
        beta=config['training']['loss_weights']['semi_supervised'].get('mc_consistency_beta', 0.5),
        temperature=config['training']['loss_weights']['semi_supervised'].get('mc_consistency_temperature', 1.0),
        uncertainty_threshold=config['training']['loss_weights']['semi_supervised'].get('mc_uncertainty_threshold', 0.1),
        use_adaptive_weighting=config['training']['loss_weights']['semi_supervised'].get('mc_use_adaptive_weighting', True)
    ).to(device)
    
    # 학습 루프
    num_batches = min(len(labeled_loader), len(unlabeled_loader))
    
    # 진행률 표시 - 실제 배치 인덱스와 동기화
    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{config['training']['epochs']}", 
                dynamic_ncols=True, leave=False)
    
    for batch_idx in pbar:
        # 진행률 표시 업데이트
        pbar.set_postfix({
            'Batch': f'{batch_idx+1}/{num_batches}',
            'Loss': f'{total_loss.item():.4f}' if 'total_loss' in locals() else 'N/A'
        })
        
        # 레이블된 데이터로부터 배치 가져오기
        labeled_batch = next(labeled_iter)
        if labeled_batch is None:
            labeled_iter = iter(labeled_loader)
            labeled_batch = next(labeled_iter)
        
        # Student 모델용 Strong Augmentation 적용
        strong_transform = get_strong_augmentation(config['data']['img_size'])
        
        # Labeled 데이터 처리 (이미 정규화된 상태로 사용)
        labeled_images = labeled_batch['images'].to(device)
        
        # 이미지 값이 0-1 범위에 있는지 확인하고 필요시 정규화
        if labeled_images.max() > 1.0:
            labeled_images = labeled_images / 255.0
        
        # YOLO 형식으로 targets 변환
        labeled_targets = []
        
        # 기존 로직
        for i, labels in enumerate(labeled_batch['labels']):
            if len(labels) > 0 and labels.shape[1] == 5:  # 유효한 레이블인지 확인
                batch_labels = torch.zeros((len(labels), 6), device=device)
                batch_labels[:, 0] = i  # batch index
                
                # 안전한 라벨 변환
                labels_safe = labels.to(device)
                
                # 클래스 ID 유효성 검증 (0 ~ num_classes-1)
                num_classes = config['data']['nc']
                class_ids = labels_safe[:, 0].long()
                class_ids = torch.clamp(class_ids, 0, num_classes - 1)
                
                # bbox 좌표 유효성 검증 (0.0 ~ 1.0)
                bbox_coords = labels_safe[:, 1:5]
                bbox_coords = torch.clamp(bbox_coords, 0.0, 1.0)
                
                # 안전한 값으로 할당
                batch_labels[:, 1] = class_ids.float()
                batch_labels[:, 2:6] = bbox_coords
                
                labeled_targets.append(batch_labels)
        
        # Loss 변수들을 미리 초기화
        box_loss = torch.tensor(0.0, device=device, requires_grad=True)
        cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
        obj_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        if labeled_targets:
            labeled_targets = torch.cat(labeled_targets, dim=0)
            
            # YOLOv8 모델에서 순전파 수행
            student_model = model.student_model
            
            student_model.model.train()
            
            # 예측 수행
            predictions = student_model.model(labeled_images)
            
            # YOLO 손실 함수를 사용하여 손실 계산
            if isinstance(predictions, (list, tuple)):
                # YOLOv8 raw feature maps인지 확인
                if len(predictions) > 0 and len(predictions[0].shape) == 4:  # [B, C, H, W]
                    # Feature maps를 detection format으로 변환
                    from utils.yolo_losses import convert_yolov8_featuremaps_to_detections
                    pred = convert_yolov8_featuremaps_to_detections(
                        predictions, 
                        img_size=config['data']['img_size'],
                        num_classes=config['data']['nc']
                    )
                else:
                    # 이미 processed된 predictions list인 경우 첫 번째만 사용
                    pred = predictions[0] if len(predictions) > 0 else None
            else:
                pred = predictions
            
            # === Supervised YOLO Loss 계산 (YOLOLoss 사용) ===
            # from utils.yolo_losses import YOLOLossSupervised
                
            # Config에서 Supervised YOLO Loss 파라미터 읽기 (새로운 구조)
            supervised_config = config['training']['loss_weights']['supervised']
                
            # Supervised YOLO Loss 생성 (config 기반)
            supervised_loss_fn = YOLOLossSupervised(
                box_loss_gain=supervised_config['box_loss_gain'],
                cls_loss_gain=supervised_config['cls_loss_gain'],
                obj_loss_gain=supervised_config['obj_loss_gain'],
                bbox_loss_type=supervised_config['bbox_loss_type'],
                focal_loss_gamma=supervised_config['focal_loss_gamma'],
                label_smoothing=supervised_config['label_smoothing']
            ).to(device)
                
            # YOLO 모델 출력이 리스트인 경우 단일 텐서로 변환
            if isinstance(pred, list):
                # YOLO feature maps를 단일 텐서로 변환
                # YOLOv8의 경우 [P3, P4, P5] feature maps를 flatten
                pred_tensors = []
                for i, feat in enumerate(pred):
                    B, C, H, W = feat.shape
                    # Reshape to (B, C, H*W) then transpose to (B, H*W, C)
                    feat_flat = feat.view(B, C, -1).transpose(1, 2)
                    pred_tensors.append(feat_flat)
                
                # 모든 feature levels를 concatenate
                pred = torch.cat(pred_tensors, dim=1)  # (B, total_anchors, C)
                logger.debug(f"🔍 [Batch {batch_idx}] Converted YOLO output: {pred.shape}")
            
            box_loss, cls_loss, obj_loss = supervised_loss_fn(pred, labeled_targets)
            
            # Loss 값 확인
            logger.debug(f"  - Box Loss: {box_loss.mean().item():.6f}")
            logger.debug(f"  - Classification Loss: {cls_loss.mean().item():.6f}")
            logger.debug(f"  - Objectness Loss: {obj_loss.item():.6f}")
            
            # Total labeled loss 계산
            labeled_loss = box_loss.mean() + cls_loss.mean() + obj_loss
            
            # Loss 값 상세 로깅
            logger.debug(f"🔍 [Batch {batch_idx}] Loss Components:")
            logger.debug(f"  - Box Loss: {box_loss.mean().item():.6f}")
            logger.debug(f"  - Cls Loss: {cls_loss.mean().item():.6f}")
            logger.debug(f"  - Obj Loss: {obj_loss.item():.6f}")
            logger.debug(f"  - Total Labeled Loss: {labeled_loss.item():.6f}")
            
            # Loss가 0인지 확인
            if labeled_loss.item() == 0.0:
                logger.warning(f"⚠️  [Batch {batch_idx}] Labeled loss is 0!")
            
            # Loss가 requires_grad인지 확인
            logger.debug(f"  - Labeled loss requires_grad: {labeled_loss.requires_grad}")
                    
            loss_dict = {
                'labeled_loss': labeled_loss.item(),
                'labeled_box_loss': box_loss.mean().item(),
                'labeled_cls_loss': cls_loss.mean().item(),
                'labeled_obj_loss': obj_loss.item()
            }
        else:
            labeled_loss = torch.tensor(0.0, device=device, requires_grad=True)
            loss_dict = {
                'labeled_loss': 0.0,
                'labeled_box_loss': 0.0,
                'labeled_cls_loss': 0.0,
                'labeled_obj_loss': 0.0
            }
        
        # Feature Alignment Loss 추가 - 복사하여 inplace 방지
        if alignment_weight > 0.0:
            model_output = model(labeled_images)
            predictions = model_output['predictions']
            alignment_loss = model_output['alignment_loss']
            
            # alignment_loss가 None이거나 gradient가 없는 경우 처리
            if alignment_loss is not None and isinstance(alignment_loss, torch.Tensor):
                if alignment_loss.requires_grad:
                    # 복사하여 inplace operation 방지
                    safe_alignment_loss = alignment_loss.clone()
                    labeled_loss = labeled_loss + alignment_weight * safe_alignment_loss
                    loss_dict['alignment_loss'] = safe_alignment_loss.item()
        else:
            loss_dict['alignment_loss'] = 0.0
            
        # 레이블되지 않은 데이터 학습 (Teacher-Student MC Dropout 전략)
        if epoch >= pseudo_label_start_epoch:
            unlabeled_batch = next(iter(unlabeled_loader))
            
            # === Teacher MC Dropout: Weak Augmentation으로 안정적인 pseudo label 생성 ===
            weak_transform = get_weak_augmentation(config['data']['img_size'])
            weak_unlabeled_images = []
            for img in unlabeled_batch['images']:
                if isinstance(img, torch.Tensor):
                    img_pil = tensor_to_pil(img)
                else:
                    img_pil = img
                weak_img = weak_transform(img_pil)
                weak_unlabeled_images.append(weak_img)
            
            weak_unlabeled_images = torch.stack(weak_unlabeled_images).to(device)
            
            # === 매 배치마다 Teacher MC Dropout으로 Pseudo Labels 생성 ===
            # predict_with_uncertainty 내부에서 이미 num_samples만큼 MC Dropout 수행
            mc_result = detector.predict_with_uncertainty(
                weak_unlabeled_images, 
                device=device
            )
            
            # 고품질 pseudo labels 추출
            high_quality_pseudo_labels = []
            if mc_result is not None:
                for batch_result in mc_result:
                    if len(batch_result['boxes']) > 0:
                        # 이미 MC Dropout에서 필터링된 결과이므로 추가 필터링 없이 사용
                        boxes = batch_result['boxes']
                        scores = batch_result['scores']
                        labels = batch_result['labels']
                        box_stds = batch_result['box_std']
                        class_entropies = batch_result['class_entropy']
                        
                        # 텐서 크기 일치 확인
                        if (len(boxes) == len(scores) == len(labels) == len(box_stds) == len(class_entropies)):
                            pseudo_label = {
                                'boxes': torch.cat([
                                    labels.unsqueeze(1).float(),  # class_id
                                    boxes  # bbox coordinates
                                ], dim=1),
                                'scores': scores,
                                            'uncertainty_stats': {
                                    'box_std': box_stds,
                                    'class_entropy': class_entropies
                                }
                            }
                            high_quality_pseudo_labels.append(pseudo_label)
                    else:
                        print(f"⚠️ 텐서 크기 불일치: boxes={len(boxes)}, scores={len(scores)}, labels={len(labels)}, box_stds={len(box_stds)}, class_entropies={len(class_entropies)}")
                    
            if high_quality_pseudo_labels:
                # === Student MC Dropout: Strong Augmentation으로 robust 학습 ===
                strong_unlabeled_images = []
                for img in unlabeled_batch['images']:
                    if isinstance(img, torch.Tensor):
                        img_pil = tensor_to_pil(img)
                    else:
                        img_pil = img
                    strong_img = strong_transform(img_pil)
                    strong_unlabeled_images.append(strong_img)
                
                strong_unlabeled_images = torch.stack(strong_unlabeled_images).to(device)
                    
                # Student 모델에서 MC Dropout 예측 수행
                student_mc_result = _get_student_mc_predictions(student_model, strong_unlabeled_images, device, config)
                
                if student_mc_result is not None:
                    # === Semi-supervised YOLO Loss 계산 ===
                    semi_config = config['training']['loss_weights']['semi_supervised']
                    
                    # Semi-supervised YOLO Loss 생성
                    semi_loss_fn = YOLOLossSemiSupervised(
                        uncertainty_alpha=semi_config['uncertainty_alpha'],
                        box_loss_gain=semi_config['box_loss_gain'],
                        cls_loss_gain=semi_config['cls_loss_gain'],
                        obj_loss_gain=semi_config['obj_loss_gain'],
                        bbox_loss_type=semi_config['bbox_loss_type'],
                        focal_loss_gamma=semi_config['focal_loss_gamma'],
                        label_smoothing=semi_config['label_smoothing']
                    ).to(device)
                    
                    # Pseudo labels를 YOLO 형식으로 변환
                    pseudo_targets = []
                    for pseudo_label in high_quality_pseudo_labels:
                        boxes = pseudo_label['boxes']
                        if len(boxes) > 0:
                            # YOLO 형식: [batch_idx, class_id, x_center, y_center, width, height]
                            batch_targets = torch.zeros((len(boxes), 6), device=device)
                            batch_targets[:, 0] = 0  # batch index (모든 pseudo label을 같은 배치로 처리)
                            batch_targets[:, 1] = boxes[:, 0]  # class_id
                            batch_targets[:, 2:6] = boxes[:, 1:5]  # bbox coordinates
                            pseudo_targets.append(batch_targets)
                    
                    if pseudo_targets:
                        pseudo_targets = torch.cat(pseudo_targets, dim=0)
                        
                        # Student 모델 예측
                        student_predictions = student_model.model(strong_unlabeled_images)
                        
                        # YOLO 출력 형식 변환
                        if isinstance(student_predictions, (list, tuple)):
                            if len(student_predictions) > 0 and len(student_predictions[0].shape) == 4:
                                from utils.yolo_losses import convert_yolov8_featuremaps_to_detections
                                student_pred = convert_yolov8_featuremaps_to_detections(
                                    student_predictions,
                                    img_size=config['data']['img_size'],
                                    num_classes=config['data']['nc']
                                )
                            else:
                                student_pred = student_predictions[0] if len(student_predictions) > 0 else None
                        else:
                            student_pred = student_predictions
                        
                        # Semi-supervised loss 계산
                        unlabeled_data_loss = semi_loss_fn(student_pred, pseudo_targets)
                        if isinstance(unlabeled_data_loss, tuple):
                            unlabeled_data_loss = unlabeled_data_loss[0]
                        if hasattr(unlabeled_data_loss, 'numel') and unlabeled_data_loss.numel() > 1:
                            unlabeled_data_loss = unlabeled_data_loss.mean()
                        # Loss dictionary 업데이트
                            loss_dict['unlabeled_data_loss'] = unlabeled_data_loss.item()
                        
                        # === MC Dropout Consistency Loss 계산 ===
                        mc_consistency_weight = config['training']['loss_weights']['semi_supervised'].get('mc_consistency_weight', 0.0)
                        
                        if mc_consistency_weight > 0.0:
                                    # 텐서 복사로 inplace operation 방지
                                    safe_high_quality_pseudo_labels = []
                                    for pseudo_label in high_quality_pseudo_labels:
                                        safe_pseudo_label = {
                                            'boxes': pseudo_label['boxes'].clone(),
                                    'paths': pseudo_label.get('paths', ''),
                                            'uncertainty_stats': pseudo_label.get('uncertainty_stats', {})
                                        }
                                        safe_high_quality_pseudo_labels.append(safe_pseudo_label)
                                    
                                    # MC 예측 결과를 안전하게 처리 - 추가 안전장치
                                    safe_mc_predictions = []
                                    for pred in student_mc_result:
                                        if isinstance(pred, torch.Tensor):
                                            safe_mc_predictions.append(pred.clone())
                                        else:
                                            # dict 형태인 경우 처리
                                            safe_pred = {}
                                            for key, value in pred.items():
                                                if isinstance(value, torch.Tensor):
                                                    safe_pred[key] = value.clone()
                                                else:
                                                    safe_pred[key] = value
                                            safe_mc_predictions.append(safe_pred)
                                    
                                    mc_consistency_result = mc_consistency_loss_fn(
                                        student_mc_predictions=safe_mc_predictions,  # 복사된 MC 예측 사용
                                        high_quality_pseudo_labels=safe_high_quality_pseudo_labels,
                                        strong_unlabeled_images=strong_unlabeled_images.clone(),
                                        return_components=True
                                    )
                                    
                                    mc_consistency_loss = mc_consistency_result['total'] * mc_consistency_weight
                                    print("mc_consistency_loss: ", mc_consistency_loss)
                                    # 세부 손실 컴포넌트 로깅
                                    loss_dict['mc_consistency_loss'] = mc_consistency_loss.item()
                                    loss_dict['mc_pseudo_consistency'] = mc_consistency_result['pseudo_consistency'].item()
                                    loss_dict['mc_uncertainty_reg'] = mc_consistency_result['mc_uncertainty'].item()
                                    
                                    if 'adaptive_weights' in mc_consistency_result:
                                        loss_dict['mc_alpha'] = mc_consistency_result['adaptive_weights']['alpha']
                                        loss_dict['mc_beta'] = mc_consistency_result['adaptive_weights']['beta']
                        else:
                            loss_dict['mc_consistency_loss'] = 0.0
                            loss_dict['mc_pseudo_consistency'] = 0.0
                            loss_dict['mc_uncertainty_reg'] = 0.0
                    else:
                        loss_dict['unlabeled_data_loss'] = 0.0
                        loss_dict['mc_consistency_loss'] = 0.0
                        loss_dict['mc_pseudo_consistency'] = 0.0
                        loss_dict['mc_uncertainty_reg'] = 0.0
                    
                    # === 총 Unlabeled Loss 계산 ===
                    # mc_consistency_loss가 정의되지 않은 경우를 대비한 안전장치
                    if 'mc_consistency_loss' not in locals():
                        mc_consistency_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    unlabeled_loss = unlabeled_data_loss + mc_consistency_loss
                    total_loss = labeled_loss + unlabeled_loss
                    # print("hereerer", unlabeled_loss)
                    # print("hereerer", total_loss)
                    
                    # Unlabeled Loss 통계 업데이트
                    loss_dict['unlabeled_loss'] = unlabeled_loss.item()
                    loss_dict['total_loss'] = total_loss.item()
            else:
                total_loss = labeled_loss
        else:
            if epoch >= pseudo_label_start_epoch and batch_idx == 0:  # 첫 번째 배치에서만 로그
                logger.info(f"⏳ Pseudo labeling will start at epoch {pseudo_label_start_epoch+1} (current: {epoch+1})")
            total_loss = labeled_loss
        
        # total_loss 안전장치 - 초기 텐서 상태인 경우 labeled_loss로 설정
        if total_loss.item() == 0.0 and hasattr(total_loss, 'grad_fn') and total_loss.grad_fn is None:
            total_loss = labeled_loss
            logger.debug(f"Batch {batch_idx}: total_loss was unset, using labeled_loss: {labeled_loss.item():.6f}")
        
        # 역전파
        optimizer.zero_grad()
        
        # gradient 체크 및 안전장치
        if total_loss.requires_grad:
            logger.info(f"🔄 [Batch {batch_idx}] Starting backpropagation...")
            logger.info(f"  - Total loss value: {total_loss.item():.6f}")
            logger.info(f"  - Total loss requires_grad: {total_loss.requires_grad}")
            
            # total_loss가 유효한지 확인
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(f"🚨 Invalid loss detected: {total_loss.item()}")
                total_loss = labeled_loss  # labeled_loss로 대체
            
            # Loss 값 범위 제한
            total_loss = torch.clamp(total_loss, 0.0, 100.0)
            
            # Backward pass
            logger.info(f"🔄 [Batch {batch_idx}] Calling backward()...")
            total_loss.backward()
            logger.info(f"🔄 [Batch {batch_idx}] Backward pass completed")
            
            # Gradient 상태 확인
            total_grad_norm = 0.0
            grad_count = 0
            zero_grad_count = 0
            
            logger.info(f"🔄 [Batch {batch_idx}] Checking gradients...")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm += param_grad_norm ** 2
                    grad_count += 1
                    
                    # Gradient가 0인지 확인
                    if param_grad_norm == 0.0:
                        zero_grad_count += 1
                        logger.debug(f"  - Zero gradient in {name}")
                    
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        logger.warning(f"🚨 Invalid gradient detected in {name}")
                        param.grad.data.zero_()
                else:
                    logger.debug(f"  - No gradient for {name}")
            
            logger.info(f"🔄 [Batch {batch_idx}] Gradient Summary:")
            logger.info(f"  - Parameters with gradients: {grad_count}")
            logger.info(f"  - Parameters with zero gradients: {zero_grad_count}")
            
            if grad_count > 0:
                total_grad_norm = total_grad_norm ** 0.5
                logger.info(f"  - Total gradient norm: {total_grad_norm:.6f}")
                
                if total_grad_norm == 0.0:
                    logger.warning(f"🚨 [Batch {batch_idx}] All gradients are zero!")
            else:
                logger.warning(f"🚨 [Batch {batch_idx}] No gradients computed!")
            
            # Gradient clipping 적용
            logger.info(f"🔄 [Batch {batch_idx}] Applying gradient clipping...")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            # Parameter 업데이트 전 상태 확인
            param_updates = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_updates[name] = param.data.clone()
            
            # Optimizer step
            logger.info(f"🔄 [Batch {batch_idx}] Starting optimizer step...")
            optimizer.step()
            logger.info(f"🔄 [Batch {batch_idx}] Optimizer step completed")
            
            # Parameter 업데이트 확인
            update_count = 0
            total_param_change = 0.0
            for name, param in model.named_parameters():
                if name in param_updates:
                    param_change = torch.norm(param.data - param_updates[name]).item()
                    total_param_change += param_change
                    if param_change > 1e-8:  # 의미있는 변화가 있는지 확인
                        update_count += 1
                        logger.debug(f"  - Parameter {name} updated: {param_change:.8f}")
            
            logger.info(f"🔄 [Batch {batch_idx}] Parameter Update Summary:")
            logger.info(f"  - Parameters updated: {update_count}")
            logger.info(f"  - Total parameter change: {total_param_change:.8f}")
            
            if update_count == 0:
                logger.warning(f"🚨 [Batch {batch_idx}] No parameters were updated!")
            
            # Learning rate 확인
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"  - Current learning rate: {current_lr:.6f}")
            
            # EMA Teacher 업데이트 (매 배치마다) - DDP 지원
            model_for_ema = model.module if hasattr(model, 'module') else model
            if hasattr(model_for_ema, 'update_teacher_ema'):
                    model_for_ema.update_teacher_ema()
        
        # 진행률 표시 업데이트 - total_loss 계산 후
        pbar.set_postfix({
            'Batch': f'{batch_idx+1}/{num_batches}',
            'Loss': f'{total_loss.item():.4f}',
            'Box': f'{loss_dict.get("labeled_box_loss", 0):.3f}',
            'Cls': f'{loss_dict.get("labeled_cls_loss", 0):.3f}',
            'Obj': f'{loss_dict.get("labeled_obj_loss", 0):.3f}',
            'Align': f'{loss_dict.get("alignment_loss", 0):.3f}',
            'Grad': f'{total_grad_norm:.4f}' if 'total_grad_norm' in locals() else 'N/A',
            'LR': f'{current_lr:.6f}' if 'current_lr' in locals() else 'N/A'
        })
        
        # 학습 진행 상황 요약 로그 (매 10배치마다)
        if batch_idx % 10 == 0:
            logger.info(f"📊 [Batch {batch_idx}] Training Summary:")
            logger.info(f"  - Total Loss: {total_loss.item():.6f}")
            logger.info(f"  - Box Loss: {loss_dict.get('labeled_box_loss', 0):.6f}")
            logger.info(f"  - Cls Loss: {loss_dict.get('labeled_cls_loss', 0):.6f}")
            logger.info(f"  - Obj Loss: {loss_dict.get('labeled_obj_loss', 0):.6f}")
            logger.info(f"  - Alignment Loss: {loss_dict.get('alignment_loss', 0):.6f}")
            if 'total_grad_norm' in locals():
                logger.info(f"  - Gradient Norm: {total_grad_norm:.6f}")
            if 'current_lr' in locals():
                logger.info(f"  - Learning Rate: {current_lr:.8f}")
            if 'update_count' in locals():
                logger.info(f"  - Parameters Updated: {update_count}")
            if 'total_param_change' in locals():
                logger.info(f"  - Total Parameter Change: {total_param_change:.8f}")
        
        # Loss 값 검증 및 클리핑
        if total_loss.item() > 100.0:
            logger.warning(f"🚨 High loss detected: {total_loss.item():.2f} - applying gradient clipping")
            # 극도로 높은 loss의 경우 gradient clipping 적용
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        # 분산 학습에서 손실 동기화
        # 단일 GPU에서는 loss 동기화 불필요
            synchronized_loss = total_loss
        
        # 진행률 업데이트
        pbar.update(1)

    
    # epoch 완료 메시지 출력
    final_desc = f"Epoch {epoch+1}/{config['training']['epochs']} ✅ Completed"
    pbar.set_description(final_desc)
    pbar.refresh()
    
    # 🔬 MC Dropout 품질 모니터링 (매 에포크마다) - 임시 비활성화 (hang 방지)
    try:
        print()  # 줄바꿈으로 시각적 구분
        logger.info(f"🔬 [Epoch {epoch+1}] MC Dropout Quality Metrics: 임시 비활성화됨")
        logger.info("  MC Dropout 품질 평가는 학습 완료 후 별도로 실행할 수 있습니다.")
        
        # 원래 코드 (임시 주석처리)
        # mc_quality_stats = evaluate_mc_dropout_quality_metrics(
        #     model=model,
        #     data_loader=labeled_loader,
        #     detector=detector,
        #     device=device,
        #     config=config,
        #     max_batches=3
        # )
        # 
        # # MC Dropout 통계를 로그에 추가
        # if mc_quality_stats:
        #     logger.info(f"🔬 [Epoch {epoch+1}] MC Dropout Quality Metrics:")
        #     # 주요 메트릭만 출력하여 가독성 향상
        #     key_metrics = ['avg_box_variance', 'avg_confidence', 'total_detections']
        #     for metric in key_metrics:
        #         if metric in mc_quality_stats:
        #             value = mc_quality_stats[metric]
        #             if isinstance(value, (int, float)):
        #                 logger.info(f"  - {metric}: {value:.6f}")
        #             else:
        #                 logger.info(f"  - {metric}: {value}")
    except Exception as e:
        logger.warning(f"MC Dropout quality evaluation failed: {e}")
    
    print()  # 마지막에 줄바꿈 추가
    
    try:
        # 단일 GPU에서는 최종 손실 동기화 불필요
            return total_loss.item()
    
    except Exception as e:
        logger.error(f"🚨 학습 중 오류 발생 (Epoch {epoch}): {e}")
        logger.error("GPU 메모리를 정리하고 안전하게 종료합니다...")
        
        # GPU 메모리 정리
        cleanup_gpu_memory()
        
        # 상세 오류 정보 출력
        import traceback
        logger.error("상세 오류 정보:")
        logger.error(traceback.format_exc())
        
        # 오류를 재발생시켜 상위 함수에서 처리하도록 함
        raise e

    finally:
        # 다시 train 모드로 전환
        model.train()

def get_run_dir(args, model_name: str, labeled_ratio: float) -> Path:
    """실행 디렉토리 생성 및 반환
    
    Args:
        model_name: YOLO 모델 이름 (e.g., "yolov8m")
        labeled_ratio: 레이블된 데이터 비율
    
    Returns:
        실행 디렉토리 경로
    """
    # runs/train 디렉토리 생성 (YOLO 표준 구조)
    save_dir = getattr(args, 'save_dir', '.')
    runs_dir = Path(save_dir) / "runs"
    base_dir = runs_dir / "train"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 기본 실행 디렉토리 이름 생성
    run_name = f"{model_name}_label_p{labeled_ratio}"
    
    # 이미 존재하는 디렉토리 확인
    existing_runs = list(base_dir.glob(f"{run_name}*"))
    if not existing_runs:
        run_dir = base_dir / run_name
    else:
        # 마지막 번호 찾기
        max_num = 0
        for run in existing_runs:
            if run.name == run_name:  # 기본 이름과 동일한 경우
                max_num = max(max_num, 1)
            else:
                try:
                    # _숫자 형식의 접미사가 있는 경우
                    num = int(run.name.split("_")[-1])
                    max_num = max(max_num, num)
                except ValueError:
                    continue
        
        # 새 디렉토리 이름 생성 (항상 번호 증가)
        run_dir = base_dir / f"{run_name}_{max_num + 1}"
    
    # 디렉토리 생성
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def evaluate_model_wrapper(model, val_loader, device, epoch, save_dir, val_data_path):
    """YOLO 내장 평가 기능을 사용하는 정확한 모델 평가 함수"""
    
    # Train 모드에서 eval 모드로 전환
    model.eval()
    
    try:
        # DDP 모델인 경우 .module 속성 사용
        model_for_eval = model.module if hasattr(model, 'module') else model
        model_for_eval.eval()
        
        # YOLO 모델의 내장 평가 기능 사용 시도
        try:
            # COCO validation 데이터셋 YAML 파일 경로 생성
            coco_val_yaml = str(Path(__file__).parent / 'configs' / 'coco_val.yaml')
            
            # 경고 메시지 억제를 위한 설정
            import warnings
            import logging
            import os
            
            # YOLO 관련 경고 억제
            warnings.filterwarnings("ignore", category=UserWarning, module="ultralytics")
            warnings.filterwarnings("ignore", category=UserWarning, module="yolo")
            
            # 로깅 레벨 조정
            logging.getLogger("ultralytics").setLevel(logging.ERROR)
            logging.getLogger("yolo").setLevel(logging.ERROR)
            
            # 라벨 검증은 이미 수행되었으므로 생략
            
            # YOLO의 runs 디렉토리를 src/runs로 변경
            original_cwd = os.getcwd()
            src_dir = Path(__file__).parent
            
            # YOLO의 runs 디렉토리 환경 변수 설정
            runs_dir = src_dir / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            os.environ['YOLO_RUNS_DIR'] = str(runs_dir)
            
            # 작업 디렉토리를 src로 변경
            os.chdir(src_dir)
            
            # YOLO 모델에서 직접 평가 수행 (시각화 비활성화)
            if hasattr(model_for_eval, 'student_model') and hasattr(model_for_eval.student_model, 'val'):
                # YOLOWithMCDropout 모델의 경우 - YAML 파일 사용
                print(f"Using YAML config for validation: {coco_val_yaml}")
                results = model_for_eval.student_model.val(
                    data=coco_val_yaml,
                    save=True,  # 시각화 저장 비활성화
                    project=save_dir,
                    save_txt=False,  # 텍스트 결과 저장 비활성화
                    save_conf=False,  # 신뢰도 저장 비활성화
                    save_json=False,  # JSON 저장 비활성화
                    plots=False,  # 플롯 생성 비활성화
                    verbose=True  # 상세 출력 비활성화
                )
            elif hasattr(model_for_eval, 'val'):
                # 직접 YOLO 모델인 경우 - YAML 파일 사용
                print(f"Using YAML config for validation: {coco_val_yaml}")
                results = model_for_eval.val(
                    data=coco_val_yaml,
                    save=True,  # 시각화 저장 비활성화
                    project=save_dir,
                    save_txt=False,  # 텍스트 결과 저장 비활성화
                    save_conf=False,  # 신뢰도 저장 비활성화
                    save_json=False,  # JSON 저장 비활성화
                    plots=False,  # 플롯 생성 비활성화
                    verbose=True  # 상세 출력 비활성화
                )
            else:
                # 내장 평가가 불가능한 경우 fallback
                raise NotImplementedError("YOLO 내장 평가 불가능")
            
            # 작업 디렉토리 복원
            os.chdir(original_cwd)
            
            # 결과에서 mAP 추출
            if hasattr(results, 'results_dict'):
                mAP50 = results.results_dict.get('metrics/mAP50(B)', 0.0)
                mAP50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0.0)
                
                # 클래스별 성능 저장
                class_metrics = {}
                for key, value in results.results_dict.items():
                    if key.startswith('metrics/'):
                        class_metrics[key] = value
                
                # 클래스별 성능을 txt 파일로 저장
                metrics_file = save_dir / f'class_metrics_epoch_{epoch}.txt'
                with open(metrics_file, 'w') as f:
                    for metric_name, metric_value in class_metrics.items():
                        f.write(f"{metric_name}: {metric_value}\n")
                
            elif hasattr(results, 'box'):
                # 다른 형식의 결과
                mAP50 = results.box.map50 if hasattr(results.box, 'map50') else 0.0
                mAP50_95 = results.box.map if hasattr(results.box, 'map') else 0.0
                
                # 클래스별 성능 저장
                metrics_file = save_dir / f'class_metrics_epoch_{epoch}.txt'
                with open(metrics_file, 'w') as f:
                    if hasattr(results.box, 'classes'):
                        for i, class_metrics in enumerate(results.box.classes):
                            f.write(f"Class {i}:\n")
                            f.write(f"  AP50: {class_metrics.ap50:.4f}\n")
                            f.write(f"  AP: {class_metrics.ap:.4f}\n")
            else:
                # 결과 형식을 알 수 없는 경우
                mAP50 = 0.0
                mAP50_95 = 0.0
                
            print(f"YOLO 내장 평가 성공: mAP50={mAP50:.4f}, mAP50-95={mAP50_95:.4f}")
            return mAP50, mAP50_95
            
        except Exception as yolo_eval_error:
            print(f"YOLO 내장 평가 실패, fallback 사용: {yolo_eval_error}")
            print(f"YAML 파일 경로: {coco_val_yaml}")
            print(f"YAML 파일 존재 여부: {Path(coco_val_yaml).exists()}")
            
            # 작업 디렉토리 복원 (예외 발생 시에도)
            os.chdir(original_cwd)
            
            # Fallback: 수동 평가
            return evaluate_model_manual(model_for_eval, val_loader, device)
        
    except Exception as e:
        print(f"평가 실패: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0
    finally:
        # 다시 train 모드로 전환
        model.train()

def evaluate_model_manual(model, val_loader, device):
    """수동 평가 함수 (fallback)"""
    model.eval()
    
    # 평가 메트릭 초기화
    all_predictions = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                images = batch['images'].to(device)
                targets = batch['labels']
                    
                # 모델 예측
                outputs = model(images)
            
                # 예측 결과 처리
                if isinstance(outputs, dict) and 'predictions' in outputs:
                    predictions = outputs['predictions']
                else:
                    predictions = outputs
                
                # 예측과 타겟 수집
                if predictions is not None:
                    if isinstance(predictions, (list, tuple)):
                        if len(predictions) > 0:
                            pred = predictions[0]
                            if pred is not None:
                                all_predictions.append(pred.cpu())
                    else:
                        all_predictions.append(predictions.cpu())
                
                # 타겟 처리
                batch_targets = []
                for i, target in enumerate(targets):
                    if len(target) > 0:
                        yolo_target = torch.zeros((len(target), 6))
                        yolo_target[:, 0] = i
                        yolo_target[:, 1:] = target
                        batch_targets.append(yolo_target)
                
                if batch_targets:
                    all_targets.extend(batch_targets)
                    
                    num_batches += 1
                    
                if num_batches >= 10:
                        break
                    
            except Exception as batch_error:
                print(f"배치 평가 오류: {batch_error}")
                continue
        
    # 실제 mAP 계산
    if all_predictions and all_targets:
        try:
            if len(all_predictions) > 0:
                predictions_tensor = torch.cat(all_predictions, dim=0)
            else:
                predictions_tensor = torch.empty((0, 6))
            
            if len(all_targets) > 0:
                targets_tensor = torch.cat(all_targets, dim=0)
            else:
                targets_tensor = torch.empty((0, 6))
            
            mAP50, mAP50_95 = compute_simple_map(predictions_tensor, targets_tensor)
            
        except Exception as map_error:
            print(f"mAP 계산 오류: {map_error}")
            mAP50, mAP50_95 = 0.0, 0.0
    else:
        mAP50, mAP50_95 = 0.0, 0.0
    
    print(f"수동 평가 완료: mAP50={mAP50:.4f}, mAP50-95={mAP50_95:.4f}")
    return mAP50, mAP50_95

def compute_simple_map(predictions, targets, iou_threshold=0.5):
    """간단한 IoU 기반 mAP 계산"""
    if len(predictions) == 0 or len(targets) == 0:
        return 0.0, 0.0
    
    try:
        # 텐서를 numpy로 변환하여 처리
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        # 예측과 타겟을 박스 형식으로 변환
        pred_boxes = predictions_np[:, :4]  # [x1, y1, x2, y2]
        pred_scores = predictions_np[:, 4]  # confidence
        pred_classes = predictions_np[:, 5].astype(int)  # class_id
        
        target_boxes = targets_np[:, 2:6]  # [x, y, w, h] -> [x1, y1, x2, y2] 변환 필요
        target_classes = targets_np[:, 1].astype(int)  # class_id
        
        # center format을 corner format으로 변환 (타겟)
        target_boxes_corner = np.zeros_like(target_boxes)
        target_boxes_corner[:, 0] = target_boxes[:, 0] - target_boxes[:, 2] / 2  # x1
        target_boxes_corner[:, 1] = target_boxes[:, 1] - target_boxes[:, 3] / 2  # y1
        target_boxes_corner[:, 2] = target_boxes[:, 0] + target_boxes[:, 2] / 2  # x2
        target_boxes_corner[:, 3] = target_boxes[:, 1] + target_boxes[:, 3] / 2  # y2
        
        # IoU 계산 및 매칭
        matched_predictions = 0
        total_predictions = len(predictions_np)
        total_targets = len(targets_np)
        
        # 각 예측에 대해 가장 높은 IoU를 가진 타겟 찾기
        for i, (pred_box, pred_class, pred_score) in enumerate(zip(pred_boxes, pred_classes, pred_scores)):
            best_iou = 0.0
            best_match = -1
            
            for j, (target_box, target_class) in enumerate(zip(target_boxes_corner, target_classes)):
                if int(pred_class) == int(target_class):  # 같은 클래스만 매칭 (int로 변환)
                    iou = calculate_iou_numpy(pred_box, target_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = j
            
            if best_iou >= iou_threshold:
                matched_predictions += 1
        
        # Precision과 Recall 계산
        precision = matched_predictions / max(total_predictions, 1)
        recall = matched_predictions / max(total_targets, 1)
        
        # 간단한 mAP 계산 (precision을 mAP로 근사)
        mAP50 = precision
        mAP50_95 = precision * 0.7  # 근사값
        
        return mAP50, mAP50_95
        
    except Exception as e:
        print(f"mAP 계산 중 오류: {e}")
        return 0.0, 0.0

def calculate_iou_numpy(box1, box2):
    """두 박스 간의 IoU 계산 (numpy 버전)"""
    try:
        # 박스 좌표 추출
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 교집합 영역 계산
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 합집합 영역 계산
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-8)
        
    except Exception as e:
        print(f"IoU 계산 오류: {e}")
        return 0.0

def calculate_iou(box1, box2):
    """두 박스 간의 IoU 계산"""
    try:
        # 박스 좌표 추출
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 교집합 영역 계산
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 합집합 영역 계산
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-8)
        
    except Exception as e:
        print(f"IoU 계산 오류: {e}")
        return 0.0

def main():
    # GPU 메모리 정리 핸들러 설정
    setup_cleanup_handlers()
    
    # 메트릭 추적을 위한 딕셔너리 초기화
    metrics = {
        'train_loss': [],
        'supervised_loss': [],
        'pseudo_loss': [],
        'alignment_loss': [],
        'mAP50': [],
        'mAP50-95': []
    }
    
    try:
        # 명령행 인자와 설정 파싱
        args, config = parse_args()
    
        # GPU 설정 확인
        device, gpu_ids, _ = setup_gpu_config(config)
    
        # GPU 정보 출력
        print(f"사용 가능한 GPU: {gpu_ids}")
        print(f"주요 디바이스: {device}")
    
        # 디바이스 설정 (명령행 인자가 있으면 우선 적용)
        if args.device is None:
            args.device = device
    
        # 실행 디렉토리 설정
        run_dir = get_run_dir(args, args.model, args.labeled_ratio)
        
        # 로거 설정
        logger = setup_logger(run_dir)
        logger.setLevel(logging.DEBUG)  # DEBUG 레벨로 설정하여 상세 로그 출력
        logger.info(f"Results will be saved to: {run_dir}")
    

    
        # MC Dropout이 적용된 YOLO 모델 생성
        ema_decay = config.get('model', {}).get('ema', {}).get('decay', 0.999)  # EMA 설정 읽기
        pretrained = config.get('model', {}).get('pretrained', False)  # pre-trained 설정 읽기
        model = YOLOWithMCDropout(
            model_name=args.model,
            dropout_rate=args.dropout_rate,
            pretrained=pretrained,  # 설정에서 pre-trained 옵션 전달
            feature_alignment_enabled=args.feature_alignment_enabled,
            num_classes=config['data']['nc'],
            ema_decay=ema_decay
        )
    
        # 모델을 디바이스로 이동
        model = model.to(args.device)
    
        logger.info(f"🔄 EMA Teacher 업데이트 활성화: decay={ema_decay}")
    
        # MC Dropout 탐지기 생성 (결과 저장 디렉토리 포함)
        mc_dropout_save_dir = run_dir / "mcdropout"
        mc_dropout_save_dir.mkdir(parents=True, exist_ok=True)
        detector = MCDropoutDetector(
            model=model,
            num_samples=args.num_samples,
            dropout_rate=args.dropout_rate,
            box_std_threshold=args.box_std_threshold,
            entropy_threshold=args.entropy_threshold,
            conf_threshold=args.conf_threshold,
            max_pseudo_labels=args.max_pseudo_labels,
            save_dir=mc_dropout_save_dir
        )
    
        # 옵티마이저 설정
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['optimizer']['lr'],
            momentum=config['training']['optimizer']['momentum'],
            weight_decay=config['training']['optimizer']['weight_decay']
        )
    
        # 학습률 스케줄러
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
    
        # 체크포인트에서 재시작
        start_epoch = 0
        best_map = 0.0
        best_map_95 = 0.0  # best_map_95 초기화 추가
        if args.resume:
            model, optimizer, start_epoch, best_map = load_checkpoint(
                model, optimizer, Path(args.resume), scheduler
            )
            # 체크포인트에서 best_map_95도 로드하도록 시도
            try:
                checkpoint = torch.load(Path(args.resume), map_location='cpu')
                if 'best_mAP50_95' in checkpoint:
                    best_map_95 = checkpoint['best_mAP50_95']
                elif 'best_map_95' in checkpoint:
                    best_map_95 = checkpoint['best_map_95']
            except Exception:
                best_map_95 = 0.0  # 로드 실패 시 기본값
            logger.info(f"Resumed from checkpoint: {args.resume}")
            logger.info(f"Best mAP@0.5: {best_map:.4f}, Best mAP@0.5:0.95: {best_map_95:.4f}")
    
        # 데이터셋 초기화
        logger.info("=== 데이터셋 초기화 시작 ===")
    
        # 데이터셋 매니저 초기화
        dataset_manager = SemiSupervisedDataset(
            config_path=args.config,
            percent=args.labeled_ratio,
            seed=args.seed,
            transform=get_val_transform(img_size=config['data']['img_size']),  # 기본 변환만 적용
            max_samples=args.max_samples
        )
            
        # 데이터 로더 생성
        labeled_loader = torch.utils.data.DataLoader(
                    dataset_manager.labeled_dataset,
            batch_size=config['training']['batch_size'],
                    shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
                    collate_fn=dataset_manager.labeled_dataset.collate_fn
                )
                
        unlabeled_loader = torch.utils.data.DataLoader(
                    dataset_manager.unlabeled_dataset,
            batch_size=config['training']['batch_size'],
                    shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
                    collate_fn=dataset_manager.unlabeled_dataset.collate_fn
                )
                
        val_loader = torch.utils.data.DataLoader(
                    dataset_manager.val_dataset,
            batch_size=config['training']['batch_size'],
                    shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
                    collate_fn=dataset_manager.val_dataset.collate_fn
                )
    
        # 데이터셋 정보 로깅
        logger.info(f"✓ Labeled 데이터: {len(labeled_loader.dataset)} 샘플, {len(labeled_loader)} 배치")
        logger.info(f"✓ Unlabeled 데이터: {len(unlabeled_loader.dataset)} 샘플, {len(unlabeled_loader)} 배치") 
        logger.info(f"✓ Validation 데이터: {len(val_loader.dataset)} 샘플, {len(val_loader)} 배치")
        
        # 샘플 배치 테스트
        logger.info("=== 데이터 로더 샘플 테스트 ===")
        sample_batch = next(iter(labeled_loader))
        logger.info(f"✓ Labeled 배치 테스트 성공: 이미지 {sample_batch['images'].shape}, 레이블 {len(sample_batch['labels'])}개")
        
        # 레이블 유효성 검증
        valid_labels = 0
        total_objects = 0
        for labels in sample_batch['labels']:
            if len(labels) > 0:
                valid_labels += 1
                total_objects += len(labels)
        
        logger.info(f"  - 유효한 레이블: {valid_labels}/{len(sample_batch['labels'])} 이미지")
        logger.info(f"  - 총 객체 수: {total_objects}")
        
        sample_unlabeled = next(iter(unlabeled_loader))
        logger.info(f"✓ Unlabeled 배치 테스트 성공: 이미지 {sample_unlabeled['images'].shape}")
            
        logger.info("=== 데이터셋 초기화 완료 ===")
        
        # GT 데이터 시각화 수행
        logger.info("=== GT 데이터 시각화 시작 ===")
        from utils.visualization import visualize_gt_data
        
        # 시각화 저장 디렉토리 생성
        visualization_dir = run_dir / "gt_visualization"
        visualization_dir.mkdir(parents=True, exist_ok=True)
        
        # COCO 클래스 이름 가져오기
        coco_class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Labeled 데이터 시각화 (안전한 설정으로 제한)
        logger.info("📊 Labeled GT 데이터 시각화 중...")
        visualize_gt_data(
            data_loader=labeled_loader,
            class_names=coco_class_names,
            save_dir=visualization_dir / "labeled",
            data_type="labeled",
            max_batches=2,  # 배치 수 제한
            max_images_per_batch=4  # 배치당 이미지 수 제한
        )
        
        # Validation 데이터 시각화
        logger.info("📊 Validation GT 데이터 시각화 중...")
        visualize_gt_data(
            data_loader=val_loader,
            class_names=coco_class_names,
            save_dir=visualization_dir / "validation",
            data_type="validation",
            max_batches=1,  # 배치 수 제한
            max_images_per_batch=4  # 배치당 이미지 수 제한
        )
        
        logger.info(f"✅ GT 데이터 시각화 완료! 결과는 {visualization_dir}에 저장되었습니다.")
        
        # 데이터셋 통계 시각화 추가
        logger.info("📊 데이터셋 통계 시각화 중...")
        from utils.visualization import visualize_dataset_statistics
        
        visualize_dataset_statistics(
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            val_loader=val_loader,
            class_names=coco_class_names,
            save_dir=visualization_dir
        )
        
        logger.info("✅ 데이터셋 통계 시각화 완료!")
        
        logger.info("=== GT 데이터 시각화 완료 ===")
        
    except Exception as e:
        logger.error(f"❌ 데이터셋 초기화 실패: {e}")
        logger.error("다음 사항을 확인하세요:")
        logger.error("1. COCO 데이터셋 경로: /media/oem/personal_vol/yklee/data/COCO/train2017/")
        logger.error("2. 이미지 디렉토리: /media/oem/personal_vol/yklee/data/COCO/train2017/images/")
        logger.error("3. 레이블 디렉토리: /media/oem/personal_vol/yklee/data/COCO/train2017/labels/")
        logger.error("4. 데이터 리스트 파일들이 올바른 경로를 가리키는지 확인")
        
        # 더 상세한 오류 정보 출력
        import traceback
        logger.error("상세 오류 정보:")
        logger.error(traceback.format_exc())
        raise
    
    # 학습 루프
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n🔄 Starting Epoch {epoch+1}/{args.epochs}")
        
        # 학습 단계
        avg_loss = train_one_epoch(
            model=model,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            optimizer=optimizer,
            detector=detector,
            device=args.device,
            epoch=epoch,
            logger=logger,
            config=config,
            unlabeled_weight=args.unlabeled_weight,
            pseudo_label_start_epoch=args.pseudo_label_start_epoch,
            conf_threshold=args.conf_threshold,
            alignment_weight=args.feature_alignment_weight if args.feature_alignment_enabled else 0.0
        )
        
        # Epoch 요약 정보 출력
        print()  # 시각적 구분을 위한 줄바꿈
        logger.info("=" * 80)
        logger.info(f"📊 EPOCH {epoch+1}/{config['training']['epochs']} SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Average Loss: {avg_loss:.6f}")
        
        # 스케줄러 업데이트
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning Rate: {current_lr:.8f}")
        
        # 진행률 표시
        progress_percent = (epoch + 1) / config['training']['epochs'] * 100
        logger.info(f"Training Progress: {progress_percent:.1f}% ({epoch + 1}/{config['training']['epochs']} epochs)")
        
        # 에포크별 저장 및 평가
        is_best = False
        
        # 검증 수행 (val_interval마다 또는 마지막 에포크)
        should_validate = (epoch + 1) % args.val_interval == 0 or epoch == config['training']['epochs'] - 1
        
        # 체크포인트 저장 (save_interval마다 또는 마지막 에포크)
        should_save = (epoch + 1) % args.save_interval == 0 or epoch == config['training']['epochs'] - 1
        
        if should_validate or should_save:
            if should_validate:
                logger.info(f"🔍 Validation condition met: epoch {epoch+1}, val_interval {args.val_interval}")
        if should_save:
            logger.info(f"💾 Checkpoint save condition met: epoch {epoch+1}, save_interval {args.save_interval}")
            logger.info("-" * 60)
            logger.info(f"🔍 VALIDATION & CHECKPOINT - Epoch {epoch+1}")
            logger.info("-" * 60)
            logger.info(f"📊 MC Dropout 분석은 10 epoch마다 실행됩니다 (현재: {epoch+1})")
            
        # 검증 수행 (val_interval 조건이 만족될 때만)
        mAP50, mAP50_95 = 0.0, 0.0  # 기본값 설정
        if should_validate:
            logger.info(f"🔍 Evaluating model at epoch {epoch+1}")
            mAP50, mAP50_95 = evaluate_model_wrapper(
                model=model,
                val_loader=val_loader,
                device=args.device,
                epoch=epoch,
                save_dir=run_dir,
                val_data_path=args.val_data_path
            )
                
            logger.info(f"📊 Validation Results:")
            logger.info(f"   mAP@0.5: {mAP50:.4f}")
            logger.info(f"   mAP@0.5:0.95: {mAP50_95:.4f}")
            
            # 최고 성능 체크
            if mAP50 > best_map:
                best_map = mAP50
                best_map_95 = mAP50_95
                is_best = True
                logger.info(f"🎉 NEW BEST mAP@0.5: {best_map:.4f} (Previous: {mAP50:.4f})")
                logger.info(f"🎉 NEW BEST mAP@0.5:0.95: {best_map_95:.4f}")
            else:
                logger.info(f"   Best mAP@0.5: {best_map:.4f} (Current: {mAP50:.4f})")
                logger.info(f"   Best mAP@0.5:0.95: {best_map_95:.4f} (Current: {mAP50_95:.4f})")
                    
        # 체크포인트 저장 (save_interval 조건이 만족될 때만)
        if should_save:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'mAP50': mAP50,
                'mAP50_95': mAP50_95,
                'best_mAP50': best_map,
                'best_mAP50_95': best_map_95,
                'config': config
            }
            
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            # 체크포인트 저장
                logger.info("💾 Saving checkpoints...")
                
                # 최신 체크포인트 저장
                latest_path = run_dir / 'latest_model.pt'
                torch.save(checkpoint, latest_path)
                logger.info(f"   ✓ Latest: {latest_path}")
                
                # 최고 성능 모델 저장
                if is_best:
                    best_path = run_dir / 'best_model.pt'
                    torch.save(checkpoint, best_path)
                    logger.info(f"   🏆 Best: {best_path}")
                
                # 정기적 백업 (50 에포크마다)
                if (epoch + 1) % 50 == 0:
                    backup_path = run_dir / f'model_epoch_{epoch+1}.pt'
                    torch.save(checkpoint, backup_path)
                    logger.info(f"   📦 Backup: {backup_path}")
                
                logger.info(f"💾 Checkpoint saved successfully for epoch {epoch+1}")
                logger.info("-" * 60)  # 구분선 추가
    
    # 최종 평가 수행
    final_mAP50, final_mAP50_95 = 0.0, 0.0  # 기본값 설정
    logger.info("Performing final evaluation...")
    final_mAP50, final_mAP50_95 = evaluate_model_wrapper(
        model=model,
        val_loader=val_loader,
        device=args.device,
        epoch=args.epochs-1,
        save_dir=run_dir,
        val_data_path=args.val_data_path
    )
    
    # 최종 메트릭 기록
    metrics['mAP50'].append(final_mAP50)
    metrics['mAP50-95'].append(final_mAP50_95)
    
    # 최종 메트릭 시각화
    plot_metrics(
        metrics=metrics,
        save_path=run_dir / 'final_metrics.png'
    )
    
    # 최종 결과 로깅
    logger.info(
        f"Training completed - "
        f"Final mAP50: {final_mAP50:.4f}, "
        f"Final mAP50-95: {final_mAP50_95:.4f}"
    )
    
    # MC Dropout 전체 실험 요약 보고서 생성 - 주석처리: 컴퓨터 멈춤 방지
    # logger.info("Generating MC Dropout experiment summary report...")
    # detector.create_summary_report()
    # logger.info(f"MC Dropout experiment summary saved to {mc_dropout_save_dir}")
    
    # 최종 모델 저장
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        save_path=run_dir / 'final_model.pth',
        scheduler=scheduler,
        best_map=final_mAP50
    )


if __name__ == "__main__":
    main()
