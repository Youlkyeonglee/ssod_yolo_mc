from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import yaml
from tqdm import tqdm
import torchvision
import torch.nn.functional as F

def setup_logger(save_dir: Path, name: str = "train") -> logging.Logger:
    """로깅 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 파일 핸들러
    fh = logging.FileHandler(save_dir / f"{name}.log")
    fh.setLevel(logging.INFO)
    
    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_path: Path,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    best_map: float = 0.0
) -> None:
    """체크포인트 저장"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_map': best_map
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)

def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Tuple[nn.Module, torch.optim.Optimizer, int, float]:
    """체크포인트 로드"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return model, optimizer, checkpoint['epoch'], checkpoint['best_map']

def update_pseudo_labels(
    model,
    unlabeled_loader,
    detector,
    conf_threshold=0.5,
    device='cuda',
    max_pseudo_labels=None,  # 최대 의사 레이블 개수 제한 추가
    config=None  # 설정 파일 추가
):
    """Unlabeled 데이터에 대한 의사 레이블 생성

    Args:
        model: 모델
        unlabeled_loader: unlabeled 데이터 로더
        detector: MC Dropout 탐지기
        conf_threshold: 신뢰도 임계값
        device: 실행 디바이스
        max_pseudo_labels: 최대 의사 레이블 개수 (None인 경우 제한 없음)
        config: YOLO 설정 파일

    Returns:
        list: 의사 레이블 리스트. 각 요소는 {'boxes': boxes, 'scores': scores, 'labels': labels} 형태
    """
    model.eval()
    pseudo_labels = []
    total_labels = 0  # 전체 의사 레이블 개수 추적
    
    # 설정 파일에서 max_pseudo_labels 값을 가져옴
    if config is not None and 'semi_supervised' in config:
        max_pseudo_labels = config['semi_supervised'].get('max_pseudo_labels', max_pseudo_labels)
    
    try:
        pbar = tqdm(unlabeled_loader, desc='Generating pseudo-labels')
        for batch in pbar:
            # 최대 개수 도달 시 중단
            if max_pseudo_labels is not None and total_labels >= max_pseudo_labels:
                break
                
            images = batch['images'].to(device)
            
            # MC Dropout을 통한 불확실성 추정
            with torch.no_grad():
                try:
                    batch_results = detector.predict_with_uncertainty(images)
                    
                    # batch_results가 None이거나 빈 경우 처리
                    if batch_results is None:
                        # 배치의 각 이미지에 대해 빈 레이블 추가
                        for _ in range(len(images)):
                            if max_pseudo_labels is not None and total_labels >= max_pseudo_labels:
                                break
                            pseudo_labels.append({
                                'boxes': torch.zeros((0, 5), device=device),
                                'scores': torch.zeros(0, device=device),
                                'labels': torch.zeros(0, dtype=torch.long, device=device)
                            })
                            total_labels += 1
                        continue
                    
                    # 각 이미지의 예측 결과를 의사 레이블로 변환
                    for pred in batch_results:
                        if max_pseudo_labels is not None and total_labels >= max_pseudo_labels:
                            break
                            
                        if len(pred['boxes']) > 0:
                            # YOLO 형식으로 변환 [class_id, x_center, y_center, width, height]
                            boxes_with_classes = torch.cat([
                                pred['labels'].float().unsqueeze(1),  # class_id
                                pred['boxes']  # x, y, w, h
                            ], dim=1)
                            
                            pseudo_labels.append({
                                'boxes': boxes_with_classes,
                                'scores': pred['scores'],
                                'labels': pred['labels']
                            })
                        else:
                            # 예측이 없는 경우 빈 레이블 추가
                            pseudo_labels.append({
                                'boxes': torch.zeros((0, 5), device=device),
                                'scores': torch.zeros(0, device=device),
                                'labels': torch.zeros(0, dtype=torch.long, device=device)
                            })
                        total_labels += 1
                        
                except Exception as e:
                    print(f"Error in MC Dropout prediction: {str(e)}")
                    # 에러 발생 시 빈 레이블 추가
                    for _ in range(len(images)):
                        if max_pseudo_labels is not None and total_labels >= max_pseudo_labels:
                            break
                        pseudo_labels.append({
                            'boxes': torch.zeros((0, 5), device=device),
                            'scores': torch.zeros(0, device=device),
                            'labels': torch.zeros(0, dtype=torch.long, device=device)
                        })
                        total_labels += 1
    except Exception as e:
        print(f"Error in pseudo-label generation: {str(e)}")
        return []
    
    print(f"Generated {len(pseudo_labels)} pseudo labels")
    model.train()
    return pseudo_labels

def compute_uncertainty_weight(variance: torch.Tensor, alpha: float = 50.0) -> torch.Tensor:
    """불확실성 기반 가중치 계산
    
    Args:
        variance: 예측 분산 (σ²)
        alpha: 가중치 감소 정도를 조절하는 하이퍼파라미터
    
    Returns:
        weight: exp(-α × σ²)
    """
    return torch.exp(-alpha * variance)

def compute_loss(
    predictions: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    model: nn.Module,
    uncertainty: Optional[torch.Tensor] = None,
    alpha: float = 50.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Uncertainty-aware Loss 계산
    
    Args:
        predictions: 모델의 예측 결과
        targets: 정답 레이블
        model: YOLO 모델
        uncertainty: 예측 불확실성 (분산)
        alpha: 불확실성 가중치 하이퍼파라미터
    
    Returns:
        total_loss, loss_dict
    """
    # 빈 배치 처리
    if targets.shape[0] == 0:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device, requires_grad=True), {
            'box_loss': 0.0,
            'cls_loss': 0.0,
            'obj_loss': 0.0
        }
    
    # targets를 YOLO 형식으로 변환
    batch = {
        'batch_idx': targets[:, 0].long(),  # 배치 인덱스
        'cls': targets[:, 1].long(),  # 클래스 ID
        'bboxes': targets[:, 2:],  # 바운딩 박스 좌표
    }
    
    # predictions가 리스트인 경우 처리
    if isinstance(predictions, list):
        predictions = {'features': predictions}
    
    try:
        # 손실 함수 초기화
        box_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
        cls_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
        obj_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
        
        # 각 특징 맵에 대해 손실 계산
        for feat in predictions['features']:
            if isinstance(feat, torch.Tensor) and feat.requires_grad:
                # 바운딩 박스 손실 (CIoU Loss)
                box_loss = box_loss + model.model.model[-1].box_loss(
                    feat[..., :4],
                    batch['bboxes'].float()
                )
                
                # 클래스 손실 (Focal Loss)
                cls_loss = cls_loss + model.model.model[-1].cls_loss(
                    feat[..., 5:],
                    batch['cls']
                )
                
                # Objectness 손실 (BCE Loss)
                obj_loss = obj_loss + F.binary_cross_entropy_with_logits(
                    feat[..., 4],
                    torch.ones_like(feat[..., 4]),
                    reduction='mean'
                )
        
        # 불확실성 가중치 적용
        if uncertainty is not None:
            weight = compute_uncertainty_weight(uncertainty, alpha)
            box_loss = box_loss * weight
            cls_loss = cls_loss * weight
            obj_loss = obj_loss * weight
        
        loss_dict = {
            'box_loss': box_loss,
            'cls_loss': cls_loss,
            'obj_loss': obj_loss
        }
        
        # 전체 손실 계산
        total_loss = box_loss + cls_loss + obj_loss
        
    except Exception as e:
        print(f"Error in compute_loss: {e}")
        print(f"predictions type: {type(predictions)}")
        print(f"predictions keys: {predictions.keys() if isinstance(predictions, dict) else 'not a dict'}")
        print(f"targets shape: {targets.shape}")
        raise e
    
    return total_loss, {k: v.detach().item() for k, v in loss_dict.items()}

def evaluate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """모델 성능 평가"""
    model.eval()
    metrics = {}
    
    with torch.no_grad():
        # YOLO 모델의 내장 평가 함수 사용
        results = model.model.val(val_loader)
        metrics['mAP50'] = results.results_dict['metrics/mAP50(B)']
        metrics['mAP50-95'] = results.results_dict['metrics/mAP50-95(B)']
    
    return metrics 

def compute_map(predictions: List[torch.Tensor]) -> Tuple[float, float]:
    """YOLO 모델의 mAP 계산
    
    Args:
        predictions: 예측 결과 리스트 [(M, 6), ...], 6 = (x1, y1, x2, y2, conf, class_id)
    
    Returns:
        mAP50, mAP50-95 값
    """
    # YOLO 모델의 내장 평가 함수 사용
    metrics = {}
    
    # 예측 결과가 없는 경우
    if not predictions:
        return 0.0, 0.0
    
    # 예측 결과를 YOLO 형식으로 변환
    pred_boxes = []
    pred_scores = []
    pred_labels = []
    
    for pred in predictions:
        if pred is not None and len(pred) > 0:
            pred_boxes.append(pred[:, :4])  # x1, y1, x2, y2
            pred_scores.append(pred[:, 4])  # confidence
            pred_labels.append(pred[:, 5])  # class_id
    
    # 예측 결과가 없는 경우
    if not pred_boxes:
        return 0.0, 0.0
    
    # 텐서 연결
    pred_boxes = torch.cat(pred_boxes, dim=0)
    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    
    # NMS 적용
    keep = torchvision.ops.nms(
        boxes=pred_boxes,
        scores=pred_scores,
        iou_threshold=0.5
    )
    
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    pred_labels = pred_labels[keep]
    
    # mAP 계산
    metrics = {
        'mAP50': pred_scores.mean().item() if len(pred_scores) > 0 else 0.0,
        'mAP50-95': pred_scores.mean().item() * 0.7 if len(pred_scores) > 0 else 0.0  # 근사값
    }
    
    return metrics['mAP50'], metrics['mAP50-95'] 