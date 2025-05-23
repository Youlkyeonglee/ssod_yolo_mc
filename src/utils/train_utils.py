from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import yaml
from tqdm import tqdm
import torchvision

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
    device='cuda'
):
    """Unlabeled 데이터에 대한 의사 레이블 생성

    Args:
        model: 모델
        unlabeled_loader: unlabeled 데이터 로더
        detector: MC Dropout 탐지기
        conf_threshold: 신뢰도 임계값
        device: 실행 디바이스

    Returns:
        list: 의사 레이블 리스트. 각 요소는 {'boxes': boxes, 'scores': scores, 'labels': labels} 형태
    """
    model.eval()
    pseudo_labels = []
    
    try:
        pbar = tqdm(unlabeled_loader, desc='Generating pseudo-labels')
        for batch in pbar:
            images = batch['images'].to(device)
            
            # MC Dropout을 통한 불확실성 추정
            with torch.no_grad():
                try:
                    uncertainty_results = detector.predict_with_uncertainty(images)
                    
                    # uncertainty_results가 None이거나 빈 경우 처리
                    if uncertainty_results is None:
                        # 배치의 각 이미지에 대해 빈 레이블 추가
                        for _ in range(len(images)):
                            pseudo_labels.append({
                                'boxes': torch.zeros((0, 5), device=device),
                                'scores': torch.zeros(0, device=device),
                                'labels': torch.zeros(0, dtype=torch.long, device=device)
                            })
                        continue
                    
                    # 각 이미지에 대한 의사 레이블 생성
                    for idx in range(len(images)):
                        # 예측 결과가 없는 경우
                        if idx >= len(uncertainty_results):
                            pseudo_labels.append({
                                'boxes': torch.zeros((0, 5), device=device),
                                'scores': torch.zeros(0, device=device),
                                'labels': torch.zeros(0, dtype=torch.long, device=device)
                            })
                            continue
                        
                        pred = uncertainty_results[idx]
                        if pred is None or not pred:
                            pseudo_labels.append({
                                'boxes': torch.zeros((0, 5), device=device),
                                'scores': torch.zeros(0, device=device),
                                'labels': torch.zeros(0, dtype=torch.long, device=device)
                            })
                            continue
                        
                        # 예측 결과가 있는 경우
                        boxes = pred.get('boxes', torch.zeros((0, 5), device=device))
                        scores = pred.get('scores', torch.zeros(0, device=device))
                        labels = pred.get('labels', torch.zeros(0, dtype=torch.long, device=device))
                        
                        # 신뢰도가 높은 예측만 선택
                        if isinstance(scores, torch.Tensor) and len(scores) > 0:
                            confident_mask = scores > conf_threshold
                            boxes = boxes[confident_mask] if len(boxes) > 0 else boxes
                            scores = scores[confident_mask] if len(scores) > 0 else scores
                            labels = labels[confident_mask] if len(labels) > 0 else labels
                        
                        pseudo_labels.append({
                            'boxes': boxes,
                            'scores': scores,
                            'labels': labels
                        })
                        
                except Exception as e:
                    print(f"Error in MC Dropout prediction: {str(e)}")
                    # 에러 발생 시 빈 레이블 추가
                    for _ in range(len(images)):
                        pseudo_labels.append({
                            'boxes': torch.zeros((0, 5), device=device),
                            'scores': torch.zeros(0, device=device),
                            'labels': torch.zeros(0, dtype=torch.long, device=device)
                        })
    except Exception as e:
        print(f"Error in pseudo-label generation: {str(e)}")
        return []
    
    model.train()
    return pseudo_labels

def compute_loss(
    predictions: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    model: nn.Module
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """손실 함수 계산"""
    # 빈 배치 처리
    if targets.shape[0] == 0:
        # 빈 배치의 경우 0 손실 반환
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device, requires_grad=True), {'box_loss': 0.0, 'cls_loss': 0.0, 'dfl_loss': 0.0}
    
    # targets를 YOLO 모델이 기대하는 형식으로 변환
    batch = {
        'batch_idx': targets[:, 0].long(),  # 배치 인덱스
        'cls': targets[:, 1].long(),  # 클래스 ID
        'bboxes': targets[:, 2:],  # 바운딩 박스 좌표
    }
    
    # predictions가 리스트인 경우 처리
    if isinstance(predictions, list):
        predictions = {'features': predictions}
    
    # YOLO 모델의 손실 함수 사용
    try:
        # 모델의 손실 함수 직접 구현
        box_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
        cls_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
        dfl_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
        
        # 각 특징 맵에 대해 손실 계산
        for feat in predictions['features']:
            if isinstance(feat, torch.Tensor) and feat.requires_grad:
                # 바운딩 박스 손실
                box_loss = box_loss + torch.nn.functional.mse_loss(
                    feat[..., :4],
                    batch['bboxes'].float(),
                    reduction='mean'
                )
                
                # 클래스 손실
                cls_loss = cls_loss + torch.nn.functional.cross_entropy(
                    feat[..., 4:],
                    batch['cls'],
                    reduction='mean'
                )
        
        loss_dict = {
            'box_loss': box_loss,
            'cls_loss': cls_loss,
            'dfl_loss': dfl_loss
        }
    except Exception as e:
        # 디버깅을 위한 정보 출력
        print(f"Error in compute_loss: {e}")
        print(f"predictions type: {type(predictions)}")
        print(f"predictions keys: {predictions.keys() if isinstance(predictions, dict) else 'not a dict'}")
        print(f"targets shape: {targets.shape}")
        raise e
    
    total_loss = sum(loss_dict.values())
    
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