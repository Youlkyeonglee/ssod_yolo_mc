import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List, Union, Dict


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate IoU between two sets of boxes.
    
    Args:
        box1: (N, 4) tensor in xyxy format
        box2: (M, 4) tensor in xyxy format
        eps: Small value to prevent division by zero
    
    Returns:
        (N, M) tensor containing IoU values
    """
    # Convert to xyxy format if needed
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unsqueeze(1).chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unsqueeze(0).chunk(4, -1)
    
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    # Union area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    
    return inter / union


def bbox_giou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate Generalized IoU loss.
    
    Args:
        box1: Predicted boxes (N, 4)
        box2: Target boxes (N, 4) 
        xywh: Whether boxes are in xywh format (True) or xyxy format (False)
        eps: Small value to prevent division by zero
    
    Returns:
        GIoU loss tensor
    """
    if xywh:
        # Convert xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1 / 2, x1 + w1 / 2, y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2 / 2, x2 + w2 / 2, y2 - h2 / 2, y2 + h2 / 2
    else:
        # Already in xyxy format
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    # Union area
    union = w1 * h1 + w2 * h2 - inter + eps
    
    # IoU
    iou = inter / union
    
    # Convex (smallest enclosing box) area
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c_area = cw * ch + eps
    
    # GIoU
    giou = iou - (c_area - union) / c_area
    return 1 - giou  # GIoU loss


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate Complete IoU loss.
    
    Args:
        box1: Predicted boxes (N, 4)
        box2: Target boxes (N, 4)
        xywh: Whether boxes are in xywh format (True) or xyxy format (False)
        eps: Small value to prevent division by zero
    
    Returns:
        CIoU loss tensor
    """
    if xywh:
        # Convert xywh to xyxy and extract center coordinates and dimensions
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1 / 2, x1 + w1 / 2, y1 - h1 / 2, y1 + h1 / 2
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2 / 2, x2 + w2 / 2, y2 - h2 / 2, y2 + h2 / 2
    else:
        # Already in xyxy format
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        x1, y1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
        x2, y2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    # Union area
    union = w1 * h1 + w2 * h2 - inter + eps
    
    # IoU
    iou = inter / union
    
    # Convex diagonal squared
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    
    # Center distance squared
    rho2 = ((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    # Aspect ratio penalty
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    alpha = v / (v - iou + (1 + eps))
    
    # CIoU
    ciou = iou - (rho2 / c2 + v * alpha)
    return 1 - ciou  # CIoU loss


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = -1, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        inputs: Predictions (N, C) where C is number of classes
        targets: Ground truth labels (N,)
        alpha: Weighting factor for rare class (default: -1 means no weighting)
        gamma: Focusing parameter
    
    Returns:
        Focal loss tensor
    """
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = (1 - pt) ** gamma * ce_loss
    
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        focal_loss = alpha_t * focal_loss
    
    return focal_loss.mean()


class YOLOLoss(nn.Module):
    """
    YOLO Loss function combining bbox, classification, and objectness losses.
    """
    
    def __init__(
        self,
        box_loss_gain: float = 7.5,
        cls_loss_gain: float = 0.5,
        obj_loss_gain: float = 1.0,
        bbox_loss_type: str = 'ciou',  # 'giou' or 'ciou'
        focal_loss_gamma: float = 0.0,  # 0.0 means standard CE loss
        label_smoothing: float = 0.0,
        eps: float = 1e-7
    ):
        super().__init__()
        self.box_loss_gain = box_loss_gain
        self.cls_loss_gain = cls_loss_gain  
        self.obj_loss_gain = obj_loss_gain
        self.bbox_loss_type = bbox_loss_type
        self.focal_loss_gamma = focal_loss_gamma
        self.label_smoothing = label_smoothing
        self.eps = eps
    
    def forward(
        self, 
        predictions: Union[torch.Tensor, List[torch.Tensor]], 
        targets: torch.Tensor,
        uncertainty_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for YOLO loss calculation.
        
        Args:
            predictions: Model predictions - can be:
                        - Tensor [B, N, 5+C] (processed detections)  
                        - List of feature maps [B, channels, H, W] (raw YOLOv8 output)
            targets: Ground truth (M, 6) where M is number of objects
                    Format: [batch_idx, class_id, x_center, y_center, width, height]
            uncertainty_weight: Optional uncertainty weighting tensor
        
        Returns:
            Tuple of (box_loss, cls_loss, obj_loss)
        """
        # print(f"🔍 YOLOLoss.forward 호출:")
        # print(f"  - predictions type: {type(predictions)}")
        # if isinstance(predictions, torch.Tensor):
        #     print(f"  - predictions shape: {predictions.shape}")
        # elif isinstance(predictions, (list, tuple)):
        #     print(f"  - predictions length: {len(predictions)}")
        #     for i, p in enumerate(predictions):
        #         print(f"    - predictions[{i}] shape: {p.shape}")
        # print(f"  - targets shape: {targets.shape}")
        # print(f"  - uncertainty_weight: {uncertainty_weight}")
        
        # 🔧 YOLOv8 feature maps 처리
        if isinstance(predictions, (list, tuple)):
            # Raw feature maps인 경우 - YOLOv8 training mode output
            if len(predictions) > 0 and len(predictions[0].shape) == 4:  # [B, C, H, W]
                # print(f"  - YOLOv8 feature maps detected, converting to detections...")
                # Feature maps를 detection format으로 변환
                predictions = convert_yolov8_featuremaps_to_detections(
                    predictions, 
                    img_size=640,  # TODO: config에서 가져오기
                    num_classes=80  # TODO: 동적으로 설정
                )
                # print(f"  - Converted predictions shape: {predictions.shape}")
            else:
                # 이미 processed된 predictions list인 경우 첫 번째만 사용
                predictions = predictions[0] if len(predictions) > 0 else torch.empty(0)
                # print(f"  - Using first prediction from list: {predictions.shape}")
        
        # 예측이 비어있거나 None인 경우 처리
        if predictions is None or (isinstance(predictions, torch.Tensor) and predictions.numel() == 0):
            # print(f"  - ❌ Empty predictions, returning zero loss")
            device = targets.device if targets.numel() > 0 else torch.device('cpu')
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, zero_loss, zero_loss
        
        # Device 확인 및 이동
        device = predictions.device
        if targets.device != device:
            targets = targets.to(device)
        
        # Batch size와 예측 차원 확인
        batch_size = predictions.shape[0]
        num_predictions = predictions.shape[1] if len(predictions.shape) > 1 else 0
        
        # print(f"  - batch_size: {batch_size}, num_predictions: {num_predictions}")
        
        if num_predictions == 0:
            # print(f"  - ❌ No predictions, returning zero loss")
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, zero_loss, zero_loss
        
        # 예측 차원 분해: [x, y, w, h, objectness, class_logits...]
        num_classes = predictions.shape[-1] - 5
        if num_classes <= 0:
            raise ValueError(f"Invalid prediction dimensions: {predictions.shape}. Expected at least 5 + num_classes.")
        
        # print(f"  - num_classes: {num_classes}")
        
        # 타겟이 없는 경우 처리
        if targets.numel() == 0:
            # print(f"  - ❌ No targets, computing objectness loss only")
            # 타겟이 없으면 objectness만 loss 계산 (모든 예측을 background로)
            obj_loss = F.binary_cross_entropy_with_logits(
                predictions[:, :, 4], 
                torch.zeros_like(predictions[:, :, 4]),
                reduction='mean'
            ) * self.obj_loss_gain
            
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, zero_loss, obj_loss
        
        # 예측과 타겟 분해
        pred_boxes = predictions[:, :, :4]  # [batch, num_pred, 4] - xywh
        pred_obj = predictions[:, :, 4]     # [batch, num_pred] - objectness  
        pred_cls = predictions[:, :, 5:]    # [batch, num_pred, num_classes] - class logits
        
        # 타겟 처리
        target_boxes = targets[:, 2:6]      # [num_targets, 4] - xywh (normalized)
        target_cls = targets[:, 1].long()   # [num_targets] - class ids
        target_batch_idx = targets[:, 0].long()  # [num_targets] - batch indices
        
        # print(f"  - pred_boxes shape: {pred_boxes.shape}")
        # print(f"  - pred_obj shape: {pred_obj.shape}")
        # print(f"  - pred_cls shape: {pred_cls.shape}")
        # print(f"  - target_boxes shape: {target_boxes.shape}")
        # print(f"  - target_cls shape: {target_cls.shape}")
        # print(f"  - target_batch_idx shape: {target_batch_idx.shape}")
        
        # Positive 샘플 할당 (간단한 버전) - 복사하여 inplace 방지
        # TODO: 더 정교한 target assignment 구현 필요
        pos_mask = torch.zeros(batch_size, num_predictions, dtype=torch.bool, device=device)
        
        if len(targets) > 0:
            # print(f"  - Processing {len(targets)} targets for positive assignment...")
            # 각 타겟에 대해 가장 가까운 예측 찾기 (단순 버전)
            for i in range(len(targets)):
                batch_idx = target_batch_idx[i]
                if 0 <= batch_idx < batch_size:
                    # IoU 기반 할당 또는 center distance 기반 할당
                    target_box = target_boxes[i:i+1].clone()  # [1, 4] - 복사
                    batch_pred_boxes = pred_boxes[batch_idx].clone()  # [num_pred, 4] - 복사
                    
                    # Center distance 계산 (간단한 방법)
                    pred_centers = batch_pred_boxes[:, :2]  # [num_pred, 2]
                    target_center = target_box[:, :2]       # [1, 2]
                    distances = torch.sum((pred_centers - target_center) ** 2, dim=1)  # [num_pred]
                    
                    # 가장 가까운 예측을 positive로 할당
                    closest_idx = torch.argmin(distances)
                    pos_mask[batch_idx, closest_idx] = True
            
            # print(f"  - Positive samples assigned: {pos_mask.sum().item()}")
        
        # Box Loss 계산 (positive 샘플에 대해서만) - 복사하여 inplace 방지
        box_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if pos_mask.any():
            # print(f"  - Computing box loss for {pos_mask.sum().item()} positive samples...")
            # positive 예측과 해당 타겟 매칭
            pos_pred_boxes = pred_boxes[pos_mask].clone()  # [num_pos, 4] - 복사
            
            # 해당하는 타겟 박스 찾기
            pos_indices = torch.where(pos_mask)
            pos_target_boxes = []
            
            for batch_idx, pred_idx in zip(pos_indices[0], pos_indices[1]):
                # 해당 배치의 타겟 중에서 매칭되는 것 찾기
                batch_targets = targets[target_batch_idx == batch_idx]
                if len(batch_targets) > 0:
                    # 첫 번째 타겟 사용 (더 정교한 매칭 필요) - 복사
                    pos_target_boxes.append(batch_targets[0, 2:6].clone())
            
            if pos_target_boxes:
                pos_target_boxes = torch.stack(pos_target_boxes)  # [num_pos, 4]
                # print(f"  - pos_pred_boxes shape: {pos_pred_boxes.shape}")
                # print(f"  - pos_target_boxes shape: {pos_target_boxes.shape}")
                
                if self.bbox_loss_type == 'giou':
                    box_loss = bbox_giou(pos_pred_boxes, pos_target_boxes, xywh=True).mean()
                elif self.bbox_loss_type == 'ciou':
                    box_loss = bbox_ciou(pos_pred_boxes, pos_target_boxes, xywh=True).mean()
                else:
                    box_loss = F.mse_loss(pos_pred_boxes, pos_target_boxes)
                
                # 안정성을 위해 loss 값을 양수로 클리핑
                box_loss = torch.clamp(box_loss, min=0.0)
                box_loss = box_loss * self.box_loss_gain
                # print(f"  - Box loss computed: {box_loss.item()}")
            else:
                # print(f"  - ❌ No matching target boxes found for positive predictions")
                pass
        else:
            # print(f"  - ❌ No positive samples, box loss = 0")
            pass
        
        # Classification Loss 계산 (positive 샘플에 대해서만) - 복사하여 inplace 방지
        cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if pos_mask.any():
            # print(f"  - Computing classification loss for {pos_mask.sum().item()} positive samples...")
            pos_pred_cls = pred_cls[pos_mask].clone()  # [num_pos, num_classes] - 복사
            
            # 해당하는 타겟 클래스 찾기
            pos_target_cls = []
            pos_indices = torch.where(pos_mask)
            
            for batch_idx, pred_idx in zip(pos_indices[0], pos_indices[1]):
                batch_targets = targets[target_batch_idx == batch_idx]
                if len(batch_targets) > 0:
                    pos_target_cls.append(batch_targets[0, 1].long().clone())  # 복사
            
            if pos_target_cls:
                pos_target_cls = torch.stack(pos_target_cls)  # [num_pos]
                # print(f"  - pos_pred_cls shape: {pos_pred_cls.shape}")
                # print(f"  - pos_target_cls shape: {pos_target_cls.shape}")
                
                if self.focal_loss_gamma > 0:
                    cls_loss = focal_loss(pos_pred_cls, pos_target_cls, gamma=self.focal_loss_gamma)
                else:
                    cls_loss = F.cross_entropy(pos_pred_cls, pos_target_cls, label_smoothing=self.label_smoothing)
                
                cls_loss = cls_loss * self.cls_loss_gain
                # print(f"  - Classification loss computed: {cls_loss.item()}")
        
        # Objectness Loss 계산 (모든 샘플에 대해) - 복사하여 inplace 방지
        obj_targets = pos_mask.float().clone()  # positive는 1, negative는 0 - 복사
        obj_loss = F.binary_cross_entropy_with_logits(
            pred_obj.clone(), obj_targets, reduction='mean'  # 예측도 복사
        ) * self.obj_loss_gain
        # print(f"  - Objectness loss computed: {obj_loss.item()}")
        
        # Uncertainty weighting 적용
        if uncertainty_weight is not None:
            # Uncertainty가 높을수록 낮은 가중치 적용
            weight = 1.0 / (1.0 + uncertainty_weight)
            box_loss = box_loss * weight.mean()
            cls_loss = cls_loss * weight.mean()
            obj_loss = obj_loss * weight.mean()
            # print(f"  - Uncertainty weighting applied: weight.mean() = {weight.mean().item()}")
        
        # print(f"✅ YOLOLoss.forward 완료:")
        # print(f"  - Final box_loss: {box_loss.item()}")
        # print(f"  - Final cls_loss: {cls_loss.item()}")
        # print(f"  - Final obj_loss: {obj_loss.item()}")
        
        return box_loss, cls_loss, obj_loss


class YOLOLossSupervised(YOLOLoss):
    """
    YOLO Loss for supervised learning.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate supervised YOLO losses.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        
        Returns:
            Tuple of (box_loss, cls_loss, obj_loss)
        """
        return super().forward(predictions, targets, uncertainty_weight=None)


class YOLOLossSemiSupervised(YOLOLoss):
    """
    YOLO Loss for semi-supervised learning with uncertainty weighting.
    """
    
    def __init__(self, uncertainty_alpha: float = 50.0, **kwargs):
        super().__init__(**kwargs)
        self.uncertainty_alpha = uncertainty_alpha
    
    def compute_uncertainty_weight(self, variance: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty-based weight: w = exp(-α × σ²)
        
        Args:
            variance: Prediction variance (σ²)
        
        Returns:
            Uncertainty weight tensor
        """
        return torch.exp(-self.uncertainty_alpha * variance)
    
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        prediction_variance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Semi-Supervised YOLO Loss 계산 (불확실성 가중치 적용)
        
        Args:
            predictions: 모델 예측 (feature map 형태 또는 detection 형태)
            targets: Pseudo label targets (YOLO 형식)
            prediction_variance: 예측 불확실성 (선택적)
        
        Returns:
            (box_loss, cls_loss, obj_loss)
        """
        # print(f"🔍 YOLOLossSemiSupervised.forward 호출:")
        # print(f"  - predictions shape: {predictions.shape}")
        # print(f"  - targets shape: {targets.shape}")
        # print(f"  - prediction_variance: {prediction_variance}")
        
        # Feature map 형태인 경우 detection format으로 변환
        if len(predictions.shape) == 4:  # [B, C, H, W] - YOLOv8 feature map
            # print(f"  - YOLOv8 feature maps detected, converting to detections...")
            predictions = convert_yolov8_featuremaps_to_detections([predictions])
            # print(f"  - Converted predictions shape: {predictions.shape}")
        
        # prediction_variance를 uncertainty_weight로 변환
        uncertainty_weight = None
        if prediction_variance is not None:
            # prediction_variance가 [N] 형태라면 targets와 매칭
            if len(prediction_variance.shape) == 1:
                uncertainty_weight = prediction_variance
            else:
                # prediction_variance가 [B, N] 형태라면 평균
                uncertainty_weight = prediction_variance.mean(dim=0)
            # print(f"  - Converted prediction_variance to uncertainty_weight: {uncertainty_weight.shape}")
        
        # 부모 클래스의 forward 호출 (uncertainty_weight 전달)
        lbox, lcls, lobj = super().forward(predictions, targets, uncertainty_weight)
        
        # print(f"✅ YOLOLossSemiSupervised.forward 완료:")
        # print(f"  - lbox: {lbox.item()}")
        # print(f"  - lcls: {lcls.item()}")
        # print(f"  - lobj: {lobj.item()}")
        
        return lbox, lcls, lobj


def create_yolo_loss(
    loss_type: str = 'supervised',
    bbox_loss_type: str = 'ciou',
    box_gain: float = 7.5,
    cls_gain: float = 0.5,
    obj_gain: float = 1.0,
    **kwargs
) -> nn.Module:
    """
    Factory function to create YOLO loss instances.
    
    Args:
        loss_type: 'supervised' or 'semi_supervised'
        bbox_loss_type: 'giou' or 'ciou'
        box_gain: Box loss weight
        cls_gain: Classification loss weight
        obj_gain: Objectness loss weight
        **kwargs: Additional arguments for specific loss types
    
    Returns:
        YOLO loss instance
    """
    common_args = {
        'box_loss_gain': box_gain,
        'cls_loss_gain': cls_gain,
        'obj_loss_gain': obj_gain,
        'bbox_loss_type': bbox_loss_type,
        **kwargs
    }
    
    if loss_type == 'supervised':
        return YOLOLossSupervised(**common_args)
    elif loss_type == 'semi_supervised':
        return YOLOLossSemiSupervised(**common_args)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# =================== 분포 기반 Consistency Loss 함수들 ===================

def kl_divergence_loss(
    p_logits: torch.Tensor, 
    q_logits: torch.Tensor, 
    temperature: float = 3.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    KL Divergence loss between two prediction distributions.
    
    Args:
        p_logits: Student strong predictions (B, N, C)
        q_logits: Student weak predictions (B, N, C)
        temperature: Temperature for softmax (higher = softer distribution)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        KL divergence loss
    """
    # Temperature scaling for softer distributions
    p_soft = F.softmax(p_logits / temperature, dim=-1)
    q_soft = F.log_softmax(q_logits / temperature, dim=-1)
    
    # KL divergence: KL(P||Q) = sum(P * log(P/Q))
    kl_loss = F.kl_div(q_soft, p_soft, reduction='none')
    
    # Temperature compensation
    kl_loss = kl_loss * (temperature ** 2)
    
    if reduction == 'mean':
        return kl_loss.mean()
    elif reduction == 'sum':
        return kl_loss.sum()
    else:
        return kl_loss


def jensen_shannon_divergence_loss(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    temperature: float = 3.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Jensen-Shannon Divergence loss (symmetric version of KL divergence).
    
    Args:
        p_logits: Student strong predictions (B, N, C)
        q_logits: Student weak predictions (B, N, C)  
        temperature: Temperature for softmax
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        JS divergence loss
    """
    # Temperature scaling
    p_soft = F.softmax(p_logits / temperature, dim=-1)
    q_soft = F.softmax(q_logits / temperature, dim=-1)
    
    # Average distribution M = (P + Q) / 2
    m = (p_soft + q_soft) / 2.0
    
    # JS divergence = [KL(P||M) + KL(Q||M)] / 2
    kl_p_m = F.kl_div(m.log(), p_soft, reduction='none')
    kl_q_m = F.kl_div(m.log(), q_soft, reduction='none')
    
    js_loss = (kl_p_m + kl_q_m) / 2.0
    
    # Temperature compensation
    js_loss = js_loss * (temperature ** 2)
    
    if reduction == 'mean':
        return js_loss.mean()
    elif reduction == 'sum':
        return js_loss.sum()
    else:
        return js_loss


def wasserstein_distance_loss(
    p_features: torch.Tensor,
    q_features: torch.Tensor,
    p: float = 2.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Wasserstein distance between feature distributions.
    
    Args:
        p_features: Strong augmentation features (B, N, D)
        q_features: Weak augmentation features (B, N, D)
        p: Order of Wasserstein distance (default: 2 for W2)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Wasserstein distance loss
    """
    # Calculate pairwise distances
    distance = torch.norm(p_features - q_features, p=p, dim=-1)
    
    if reduction == 'mean':
        return distance.mean()
    elif reduction == 'sum':
        return distance.sum()
    else:
        return distance


def cosine_similarity_loss(
    p_features: torch.Tensor,
    q_features: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Cosine similarity loss between feature vectors.
    
    Args:
        p_features: Strong augmentation features (B, N, D)
        q_features: Weak augmentation features (B, N, D)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Cosine similarity loss (1 - cosine_similarity)
    """
    # Normalize features
    p_norm = F.normalize(p_features, p=2, dim=-1)
    q_norm = F.normalize(q_features, p=2, dim=-1)
    
    # Cosine similarity
    cosine_sim = (p_norm * q_norm).sum(dim=-1)
    
    # Convert to loss (1 - similarity)
    cosine_loss = 1.0 - cosine_sim
    
    if reduction == 'mean':
        return cosine_loss.mean()
    elif reduction == 'sum':
        return cosine_loss.sum()
    else:
        return cosine_loss


def entropy_regularization_loss(
    logits: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Entropy regularization to encourage prediction diversity.
    
    Args:
        logits: Prediction logits (B, N, C)
        temperature: Temperature scaling
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Negative entropy (encourages high entropy/diversity)
    """
    # Convert to probabilities
    probs = F.softmax(logits / temperature, dim=-1)
    
    # Calculate entropy: H(p) = -sum(p * log(p))
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    
    # Return negative entropy as loss (to maximize entropy)
    entropy_loss = -entropy
    
    if reduction == 'mean':
        return entropy_loss.mean()
    elif reduction == 'sum':
        return entropy_loss.sum()
    else:
        return entropy_loss


class DistributionalConsistencyLoss(nn.Module):
    """
    분포 기반 Consistency Loss for Teacher-Student Semi-Supervised Learning.
    Strong/Weak augmentation 간의 예측 분포 일관성을 측정합니다.
    """
    
    def __init__(
        self,
        loss_type: str = 'kl_divergence',  # 'kl_divergence', 'js_divergence', 'wasserstein', 'cosine'
        temperature: float = 3.0,
        alpha: float = 0.5,  # KL + JS divergence 조합 시 가중치
        use_entropy_reg: bool = False,
        entropy_weight: float = 0.1,
        bbox_consistency_weight: float = 1.0,
        class_consistency_weight: float = 1.0,
        obj_consistency_weight: float = 1.0
    ):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.alpha = alpha
        self.use_entropy_reg = use_entropy_reg
        self.entropy_weight = entropy_weight
        self.bbox_consistency_weight = bbox_consistency_weight
        self.class_consistency_weight = class_consistency_weight
        self.obj_consistency_weight = obj_consistency_weight
    
    def forward(
        self, 
        strong_predictions: torch.Tensor, 
        weak_predictions: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Calculate distributional consistency loss.
        
        Args:
            strong_predictions: Student strong augmentation predictions (B, N, 5+C)
            weak_predictions: Student weak augmentation predictions (B, N, 5+C)
            return_components: Whether to return individual loss components
        
        Returns:
            Total consistency loss or dictionary of components
        """
        device = strong_predictions.device
        
        # Ensure same shape
        if strong_predictions.shape != weak_predictions.shape:
            min_size = min(strong_predictions.size(1), weak_predictions.size(1))
            strong_predictions = strong_predictions[:, :min_size, :]
            weak_predictions = weak_predictions[:, :min_size, :]
        
        # Extract components: [x, y, w, h, obj_conf, cls1, cls2, ...]
        # BBox consistency (coordinates)
        strong_bbox = strong_predictions[:, :, :4]  # x, y, w, h
        weak_bbox = weak_predictions[:, :, :4]
        
        # Objectness consistency
        strong_obj = strong_predictions[:, :, 4:5]  # objectness confidence
        weak_obj = weak_predictions[:, :, 4:5]
        
        # Class consistency
        strong_cls = strong_predictions[:, :, 5:]  # class logits
        weak_cls = weak_predictions[:, :, 5:]
        
        loss_components = {}
        
        # 1. BBox Consistency Loss (Wasserstein distance for coordinates)
        if self.bbox_consistency_weight > 0:
            bbox_loss = wasserstein_distance_loss(strong_bbox, weak_bbox, p=2.0)
            loss_components['bbox_consistency'] = bbox_loss * self.bbox_consistency_weight
        else:
            loss_components['bbox_consistency'] = torch.tensor(0.0, device=device)
        
        # 2. Objectness Consistency Loss (KL divergence for binary classification)
        if self.obj_consistency_weight > 0:
            # Convert objectness to probabilities
            strong_obj_prob = torch.sigmoid(strong_obj)
            weak_obj_prob = torch.sigmoid(weak_obj)
            
            # Binary KL divergence
            eps = 1e-8
            strong_obj_prob = torch.clamp(strong_obj_prob, eps, 1-eps)
            weak_obj_prob = torch.clamp(weak_obj_prob, eps, 1-eps)
            
            obj_kl = strong_obj_prob * torch.log(strong_obj_prob / weak_obj_prob) + \
                     (1 - strong_obj_prob) * torch.log((1 - strong_obj_prob) / (1 - weak_obj_prob))
            
            loss_components['obj_consistency'] = obj_kl.mean() * self.obj_consistency_weight
        else:
            loss_components['obj_consistency'] = torch.tensor(0.0, device=device)
        
        # 3. Class Consistency Loss (주요 분포 기반 loss)
        if self.class_consistency_weight > 0 and strong_cls.size(-1) > 0:
            if self.loss_type == 'kl_divergence':
                cls_loss = kl_divergence_loss(strong_cls, weak_cls, self.temperature)
            elif self.loss_type == 'js_divergence':
                cls_loss = jensen_shannon_divergence_loss(strong_cls, weak_cls, self.temperature)
            elif self.loss_type == 'kl_js_combined':
                kl_loss = kl_divergence_loss(strong_cls, weak_cls, self.temperature)
                js_loss = jensen_shannon_divergence_loss(strong_cls, weak_cls, self.temperature)
                cls_loss = self.alpha * kl_loss + (1 - self.alpha) * js_loss
            elif self.loss_type == 'cosine':
                cls_loss = cosine_similarity_loss(strong_cls, weak_cls)
            elif self.loss_type == 'wasserstein':
                cls_loss = wasserstein_distance_loss(strong_cls, weak_cls, p=2.0)
            else:
                # Fallback to MSE
                cls_loss = F.mse_loss(strong_cls, weak_cls)
            
            loss_components['class_consistency'] = cls_loss * self.class_consistency_weight
        else:
            loss_components['class_consistency'] = torch.tensor(0.0, device=device)
        
        # 4. Entropy Regularization (선택사항)
        if self.use_entropy_reg and self.entropy_weight > 0:
            strong_entropy = entropy_regularization_loss(strong_cls, self.temperature)
            weak_entropy = entropy_regularization_loss(weak_cls, self.temperature)
            entropy_loss = (strong_entropy + weak_entropy) / 2.0
            loss_components['entropy_reg'] = entropy_loss * self.entropy_weight
        else:
            loss_components['entropy_reg'] = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = sum(loss_components.values())
        
        if return_components:
            return {
                'total': total_loss,
                'components': loss_components
            }
        else:
            return total_loss


def create_distributional_consistency_loss(
    loss_type: str = 'kl_divergence',
    temperature: float = 3.0,
    **kwargs
) -> DistributionalConsistencyLoss:
    """
    Factory function for distributional consistency loss.
    
    Args:
        loss_type: Type of distributional loss
        temperature: Temperature for softmax distributions
        **kwargs: Additional arguments
    
    Returns:
        DistributionalConsistencyLoss instance
    """
    return DistributionalConsistencyLoss(
        loss_type=loss_type,
        temperature=temperature,
        **kwargs
    )


def convert_yolov8_featuremaps_to_detections(
    feature_maps: List[torch.Tensor], 
    img_size: int = 640,
    num_classes: int = 80
) -> torch.Tensor:
    """
    YOLOv8 raw feature maps를 detection format으로 변환
    
    Args:
        feature_maps: List of feature maps [batch, 144, H, W]
        img_size: Input image size
        num_classes: Number of classes
    
    Returns:
        Tensor of shape [batch, total_anchors, 5+num_classes]
        Format: [x, y, w, h, obj_conf, class_logits...]
    """
    # print(f"🔧 convert_yolov8_featuremaps_to_detections 호출:")
    # print(f"  - feature_maps 개수: {len(feature_maps)}")
    # for i, feat in enumerate(feature_maps):
    #     print(f"  - feature_maps[{i}] shape: {feat.shape}")
    # print(f"  - img_size: {img_size}, num_classes: {num_classes}")
    
    batch_size = feature_maps[0].shape[0]
    device = feature_maps[0].device
    all_predictions = []
    
    # YOLOv8 anchor strides
    strides = [8, 16, 32]  # P3, P4, P5
    
    for i, (feat, stride) in enumerate(zip(feature_maps, strides)):
        # print(f"  - Processing feature map {i} with stride {stride}:")
        b, c, h, w = feat.shape
        # print(f"    - feat shape: {feat.shape}")
        
        # YOLOv8 채널 분해: 64 (bbox DFL) + 80 (classes) = 144
        bbox_channels = 64  # 4 coordinates * 16 DFL bins
        class_channels = num_classes  # 80
        
        # print(f"    - bbox_channels: {bbox_channels}, class_channels: {class_channels}")
        
        # 채널 분리
        bbox_feat = feat[:, :bbox_channels, :, :]  # [b, 64, h, w] - DFL regression
        cls_feat = feat[:, bbox_channels:bbox_channels+class_channels, :, :]  # [b, 80, h, w] - classes
        
        # print(f"    - bbox_feat shape: {bbox_feat.shape}")
        # print(f"    - cls_feat shape: {cls_feat.shape}")
        
        # Grid 생성 (YOLOv8 방식)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # print(f"    - grid_x shape: {grid_x.shape}, grid_y shape: {grid_y.shape}")
        
        # Feature map을 [b, h, w, channels] 형태로 변환
        bbox_pred = bbox_feat.permute(0, 2, 3, 1).contiguous()  # [b, h, w, 64]
        cls_pred = cls_feat.permute(0, 2, 3, 1).contiguous()    # [b, h, w, 80]
        
        # print(f"    - bbox_pred shape: {bbox_pred.shape}")
        # print(f"    - cls_pred shape: {cls_pred.shape}")
        
        # DFL regression을 좌표로 변환
        bbox_pred = bbox_pred.view(b, h, w, 4, 16)  # [b, h, w, 4, 16]
        bbox_pred = F.softmax(bbox_pred, dim=-1)    # DFL softmax
        
        # print(f"    - bbox_pred after reshape: {bbox_pred.shape}")
        
        # DFL 가중합으로 실제 좌표 계산
        dfl_range = torch.arange(16, device=device, dtype=torch.float32)
        bbox_pred = (bbox_pred * dfl_range.view(1, 1, 1, 1, 16)).sum(dim=-1)  # [b, h, w, 4]
        
        # print(f"    - bbox_pred after DFL: {bbox_pred.shape}")
        
        # YOLOv8 좌표 변환 (DFL은 distance이므로 절대 좌표로 변환)
        # lt (left-top), rb (right-bottom) distances를 center + size로 변환
        lt = bbox_pred[..., :2]  # [b, h, w, 2] - left, top distances
        rb = bbox_pred[..., 2:]  # [b, h, w, 2] - right, bottom distances
        
        # Grid 좌표 확장
        grid_xy = torch.stack([grid_x, grid_y], dim=-1)  # [h, w, 2]
        grid_xy = grid_xy.unsqueeze(0).expand(b, h, w, 2)  # [b, h, w, 2]
        
        # Center 좌표 계산 (grid + offset)
        xy = grid_xy + 0.5  # Grid center
        
        # Box 좌표 계산 (lt, rb distances를 이용)
        x1y1 = xy - lt  # Top-left
        x2y2 = xy + rb  # Bottom-right
        
        # Center와 크기로 변환 (xywh format)
        center_x = (x1y1[..., 0] + x2y2[..., 0]) / 2
        center_y = (x1y1[..., 1] + x2y2[..., 1]) / 2
        width = x2y2[..., 0] - x1y1[..., 0]
        height = x2y2[..., 1] - x1y1[..., 1]
        
        # 정규화 (stride로 나누고 이미지 크기로 정규화)
        center_x = center_x * stride / img_size
        center_y = center_y * stride / img_size
        width = width * stride / img_size
        height = height * stride / img_size
        
        # XYWH 결합
        xywh = torch.stack([center_x, center_y, width, height], dim=-1)  # [b, h, w, 4]
        
        # print(f"    - xywh shape: {xywh.shape}")
        
        # Objectness는 클래스 최대값으로 근사 (YOLOv8는 별도 objectness 없음)
        objectness = torch.sigmoid(cls_pred.max(dim=-1, keepdim=True)[0])  # [b, h, w, 1]
        
        # print(f"    - objectness shape: {objectness.shape}")
        
        # Flatten to [b, h*w, features]
        xywh_flat = xywh.view(b, h*w, 4)
        obj_flat = objectness.view(b, h*w, 1)
        cls_flat = cls_pred.view(b, h*w, num_classes)
        
        # print(f"    - xywh_flat shape: {xywh_flat.shape}")
        # print(f"    - obj_flat shape: {obj_flat.shape}")
        # print(f"    - cls_flat shape: {cls_flat.shape}")
        
        # Final prediction: [x, y, w, h, obj, cls1, cls2, ...]
        prediction = torch.cat([xywh_flat, obj_flat, cls_flat], dim=-1)  # [b, h*w, 5+num_classes]
        all_predictions.append(prediction)
        
        # print(f"    - prediction shape: {prediction.shape}")
    
    # 모든 스케일 결합
    final_predictions = torch.cat(all_predictions, dim=1)  # [b, total_anchors, 5+num_classes]
    # print(f"✅ Final predictions shape: {final_predictions.shape}")
    return final_predictions


def calculate_unlabeled_loss(model, high_quality_pseudo_labels, strong_unlabeled_images, 
                            unlabeled_weight, device, config, logger):
    """
    Unlabeled Data Loss 계산을 위한 함수 (단순화된 구조)
    
    Args:
        model: Teacher-Student 모델
        high_quality_pseudo_labels: 고품질 pseudo label 리스트
        strong_unlabeled_images: Strong augmentation된 unlabeled 이미지
        unlabeled_weight: Unlabeled loss 가중치
        device: 디바이스
        config: 설정
        logger: 로거
    
    Returns:
        torch.Tensor: Unlabeled data loss
    """
    # print(f"🔍 calculate_unlabeled_loss 호출됨")
    # print(f"  - high_quality_pseudo_labels 개수: {len(high_quality_pseudo_labels) if high_quality_pseudo_labels else 0}")
    # print(f"  - strong_unlabeled_images shape: {strong_unlabeled_images.shape}")
    # print(f"  - unlabeled_weight: {unlabeled_weight}")
    
    try:
        # DDP 지원
        if hasattr(model, 'module'):
            student_model = model.module.student_model
        else:
            student_model = model.student_model
        
        student_model.model.train()
        
        # Student 모델로 Strong Augmentation된 unlabeled 데이터 예측
        student_predictions = student_model.model(strong_unlabeled_images)
        
        # Pseudo Label을 YOLO target 형식으로 변환 - 복사하여 inplace 방지
        pseudo_targets = []
        # print(f"🔍 high_quality_pseudo_labels 개수: {len(high_quality_pseudo_labels)}")
        
        for i, pseudo_label in enumerate(high_quality_pseudo_labels[:len(strong_unlabeled_images)]):
            # print(f"  - Pseudo label {i}: {pseudo_label.keys()}")
            if 'boxes' in pseudo_label and len(pseudo_label['boxes']) > 0:
                # print(f"    - boxes shape: {pseudo_label['boxes'].shape}")
                boxes = pseudo_label['boxes'].clone()  # 복사
                batch_labels = torch.zeros((len(boxes), 6), device=device)
                batch_labels[:, 0] = i  # batch index
                batch_labels[:, 1:] = boxes  # [class_id, x, y, w, h]
                pseudo_targets.append(batch_labels)
                # print(f"    - batch_labels shape: {batch_labels.shape}")
            else:
                # print(f"    - boxes 없음 또는 비어있음")
                pass
        
        # print(f"🔍 pseudo_targets 개수: {len(pseudo_targets)}")
        
        if pseudo_targets:
            pseudo_targets = torch.cat(pseudo_targets, dim=0)
            # print(f"📊 Pseudo targets 생성: {pseudo_targets.shape}")
            # Semi-Supervised YOLO Loss 계산
            try:
                # Config에서 Semi-Supervised YOLO Loss 파라미터 읽기
                semi_config = config['training']['loss_weights']['semi_supervised']
                # Semi-Supervised YOLO Loss 생성
                semi_loss_fn = YOLOLossSemiSupervised(
                    uncertainty_alpha=semi_config['uncertainty_alpha'],
                    box_loss_gain=semi_config['box_loss_gain'],
                    cls_loss_gain=semi_config['cls_loss_gain'],
                    obj_loss_gain=semi_config['obj_loss_gain'],
                    bbox_loss_type=semi_config['bbox_loss_type'],
                    focal_loss_gamma=semi_config['focal_loss_gamma'],
                    label_smoothing=semi_config['label_smoothing']
                ).to(device)
                
                # 불확실성 가중치 계산
                prediction_variance = None
                if high_quality_pseudo_labels:
                    variances = []
                    for pl in high_quality_pseudo_labels[:len(strong_unlabeled_images)]:
                        reliability_score = pl['uncertainty_stats'].get('reliability_score', 0.01)
                        variance = max(0.01, 1.0 - reliability_score)
                        variances.append(variance)
                    
                    if variances:
                        prediction_variance = torch.tensor(variances, device=device).mean()
                
                # Semi-Supervised Loss 계산
                if isinstance(student_predictions, (list, tuple)):
                    pred = student_predictions[0] if len(student_predictions) > 0 else None
                else:
                    pred = student_predictions
                
                if pred is not None:
                    # print(f"🔍 Loss 계산 시작:")
                    # print(f"  - pred shape: {pred.shape}")
                    # print(f"  - pseudo_targets shape: {pseudo_targets.shape}")
                    # print(f"  - prediction_variance: {prediction_variance}")
                    
                    lbox, lcls, lobj = semi_loss_fn(pred, pseudo_targets, prediction_variance)
                    # print("lbox: ", lbox)
                    # print("lcls: ", lcls)
                    # print("lobj: ", lobj)
                    unlabeled_data_loss = (lbox + lcls + lobj) * unlabeled_weight
                    # print(f"✅ Loss 계산 완료: lbox={lbox:.4f}, lcls={lcls:.4f}, lobj={lobj:.4f}")
                    return unlabeled_data_loss
                else:
                    # print(f"❌ pred가 None입니다")
                    pass
                
            except Exception as e:
                logger.debug(f"Semi-Supervised YOLO Loss calculation failed: {e}")
        
        # 실패 시 기본값 반환
        return torch.tensor(0.0, device=device, requires_grad=True)
        
    except Exception as e:
        logger.debug(f"Unlabeled Data Loss calculation failed: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True)


class MCDropoutConsistencyLoss(nn.Module):
    """
    MC Dropout 기반 Consistency Loss
    
    Student 모델의 MC Dropout 예측과 high-quality pseudo labels 간의 일관성을 학습
    불확실성 정보를 활용하여 신뢰도 기반 가중치 적용
    
    MC Consistency Loss = α * Pseudo_Label_Consistency + β * MC_Uncertainty_Regularization
    """
    
    def __init__(
        self,
        alpha: float = 1.0,          # Pseudo label consistency weight
        beta: float = 0.5,           # MC uncertainty regularization weight
        temperature: float = 1.0,    # Temperature scaling
        uncertainty_threshold: float = 0.1,  # 불확실성 임계값
        use_adaptive_weighting: bool = True  # 적응적 가중치 사용
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.uncertainty_threshold = uncertainty_threshold
        self.use_adaptive_weighting = use_adaptive_weighting
        
        # 적응적 가중치를 위한 running statistics
        self.register_buffer('running_pseudo_consistency', torch.tensor(0.0))
        self.register_buffer('running_mc_uncertainty', torch.tensor(0.0))
        self.register_buffer('num_updates', torch.tensor(0))
    
    def forward(
        self,
        student_mc_predictions: Union[List[torch.Tensor], List[dict]],  # Student MC Dropout 예측들 또는 detector 결과
        high_quality_pseudo_labels: List[dict],      # Teacher가 생성한 pseudo labels
        strong_unlabeled_images: torch.Tensor,       # Strong augmentation된 이미지
        return_components: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        MC Dropout Consistency Loss 계산
        
        Args:
            student_mc_predictions: Student 모델의 MC Dropout 예측 리스트
            high_quality_pseudo_labels: Teacher가 생성한 고품질 pseudo label 리스트
            strong_unlabeled_images: Strong augmentation된 unlabeled 이미지
            return_components: 개별 loss 컴포넌트 반환 여부
        
        Returns:
            Total MC consistency loss or dictionary of components
        """
        # detector 결과인지 확인 (dict 형태)
        if isinstance(student_mc_predictions, list) and len(student_mc_predictions) > 0:
            if isinstance(student_mc_predictions[0], dict):
                # detector 결과를 MC tensor로 변환
                mc_tensor = self._convert_detector_result_to_mc_tensor(student_mc_predictions, strong_unlabeled_images)
                if mc_tensor is None:
                    # 변환 실패 시 기본값 반환
                    device = strong_unlabeled_images.device
                    return torch.tensor(0.0, device=device, requires_grad=True)
            else:
                # 기존 MC predictions 리스트
                if len(student_mc_predictions) < 2:
                    raise ValueError("MC Consistency Loss requires at least 2 MC samples")
                # 복사하여 inplace operation 방지
                mc_tensor = torch.stack([pred.clone() for pred in student_mc_predictions], dim=0)  # (T, B, N, C)
        else:
            raise ValueError("Invalid student_mc_predictions format")
        
        device = mc_tensor.device
        T, B, N, C = mc_tensor.shape
        
        # 1. Pseudo Label Consistency Loss
        pseudo_consistency_loss = self._compute_pseudo_label_consistency(
            mc_tensor, high_quality_pseudo_labels, strong_unlabeled_images
        )
        
        # 2. MC Uncertainty Regularization Loss
        mc_uncertainty_loss = self._compute_mc_uncertainty_regularization(mc_tensor)
        
        # 3. 적응적 가중치 조정
        if self.use_adaptive_weighting:
            adaptive_weights = self._compute_adaptive_weights(
                pseudo_consistency_loss, mc_uncertainty_loss
            )
            alpha, beta = adaptive_weights
        else:
            alpha, beta = self.alpha, self.beta
        
        # 4. Total MC Consistency Loss 계산
        total_loss = alpha * pseudo_consistency_loss + beta * mc_uncertainty_loss
        
        if return_components:
            return {
                'total': total_loss,
                'pseudo_consistency': pseudo_consistency_loss,
                'mc_uncertainty': mc_uncertainty_loss,
                'adaptive_weights': {'alpha': alpha, 'beta': beta}
            }
        else:
            return total_loss
    
    def _compute_pseudo_label_consistency(
        self,
        mc_tensor: torch.Tensor,  # (T, B, N, C)
        high_quality_pseudo_labels: List[dict],
        strong_unlabeled_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Student MC 예측과 Teacher pseudo label 간의 일관성 계산
        불확실성 기반 가중치 적용
        """
        device = mc_tensor.device
        T, B, N, C = mc_tensor.shape
        
        # MC 예측의 평균과 분산 계산
        mean_pred = mc_tensor.mean(dim=0)  # (B, N, C)
        pred_variance = mc_tensor.var(dim=0)  # (B, N, C)
        
        # 불확실성 기반 신뢰도 가중치 계산
        uncertainty_weights = self._compute_uncertainty_weights(pred_variance)
        
        total_consistency_loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_pairs = 0
        
        for batch_idx in range(B):
            if batch_idx < len(high_quality_pseudo_labels):
                pseudo_label = high_quality_pseudo_labels[batch_idx]
                
                if 'boxes' in pseudo_label and len(pseudo_label['boxes']) > 0:
                    pseudo_boxes = pseudo_label['boxes']  # [num_objects, 5] (class_id, x, y, w, h)
                    
                    # Student 예측에서 해당 배치의 예측 추출
                    batch_pred = mean_pred[batch_idx]  # (N, C)
                    batch_uncertainty = uncertainty_weights[batch_idx]  # (N,)
                    
                    # Pseudo label과 Student 예측 간의 일관성 계산
                    consistency_loss = self._compute_box_class_consistency(
                        batch_pred, pseudo_boxes, batch_uncertainty
                    )
                    
                    total_consistency_loss = total_consistency_loss + consistency_loss
                    valid_pairs += 1
        
        # 평균 계산
        if valid_pairs > 0:
            return total_consistency_loss / valid_pairs
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def _compute_box_class_consistency(
        self,
        student_pred: torch.Tensor,  # (N, C)
        pseudo_boxes: torch.Tensor,  # (num_objects, 5)
        uncertainty_weights: torch.Tensor  # (N,)
    ) -> torch.Tensor:
        """
        박스와 클래스 예측 간의 일관성 계산
        """
        device = student_pred.device
        
        if len(pseudo_boxes) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Student 예측에서 박스와 클래스 분리
        student_boxes = student_pred[:, :4]  # (N, 4) - x, y, w, h
        student_obj = torch.sigmoid(student_pred[:, 4])  # (N,) - objectness
        student_cls = student_pred[:, 5:]  # (N, num_classes) - class logits
        
        # Pseudo label에서 박스와 클래스 분리
        pseudo_cls_ids = pseudo_boxes[:, 0].long()  # (num_objects,) - class IDs
        pseudo_boxes_coords = pseudo_boxes[:, 1:5]  # (num_objects, 4) - x, y, w, h
        
        # IoU 기반 매칭
        iou_matrix = self._compute_iou_matrix(student_boxes, pseudo_boxes_coords)
        matched_indices = self._match_predictions_to_pseudo(iou_matrix, student_obj)
        
        consistency_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_matches = 0
        
        for student_idx, pseudo_idx in matched_indices:
            if student_idx is not None and pseudo_idx is not None:
                # 박스 일관성 (MSE with uncertainty weight)
                box_loss = F.mse_loss(
                    student_boxes[student_idx], 
                    pseudo_boxes_coords[pseudo_idx]
                )
                
                # 클래스 일관성 (Cross-entropy with uncertainty weight)
                target_cls = pseudo_cls_ids[pseudo_idx]
                cls_loss = F.cross_entropy(
                    student_cls[student_idx].unsqueeze(0), 
                    target_cls.unsqueeze(0)
                )
                
                # 불확실성 가중치 적용
                weight = uncertainty_weights[student_idx]
                weighted_loss = weight * (box_loss + cls_loss)
                
                consistency_loss = consistency_loss + weighted_loss
                num_matches += 1
        
        if num_matches > 0:
            return consistency_loss / num_matches
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def _compute_mc_uncertainty_regularization(self, mc_tensor: torch.Tensor) -> torch.Tensor:
        """
        MC Dropout 예측의 불확실성 정규화
        너무 높은 불확실성을 방지하면서도 적절한 불확실성 유지
        """
        T, B, N, C = mc_tensor.shape
        
        # 예측 분산 계산
        pred_variance = mc_tensor.var(dim=0)  # (B, N, C)
        
        # 박스와 클래스 분리
        box_variance = pred_variance[..., :4].sum(dim=-1)  # (B, N)
        cls_variance = pred_variance[..., 5:].sum(dim=-1)  # (B, N)
        
        # 불확실성 정규화: 너무 높거나 너무 낮은 불확실성 모두 페널티
        target_uncertainty = self.uncertainty_threshold
        
        box_uncertainty_loss = F.mse_loss(box_variance, torch.full_like(box_variance, target_uncertainty))
        cls_uncertainty_loss = F.mse_loss(cls_variance, torch.full_like(cls_variance, target_uncertainty))
        
        return box_uncertainty_loss + cls_uncertainty_loss
    
    def _compute_uncertainty_weights(self, pred_variance: torch.Tensor) -> torch.Tensor:
        """
        불확실성 기반 신뢰도 가중치 계산
        불확실성이 낮을수록 높은 가중치
        """
        # 전체 분산의 평균
        total_variance = pred_variance.sum(dim=-1)  # (B, N)
        
        # 불확실성을 신뢰도로 변환 (낮은 분산 = 높은 신뢰도)
        confidence = torch.exp(-total_variance / self.temperature)
        
        # 정규화
        confidence = torch.clamp(confidence, 0.1, 1.0)
        
        return confidence
    
    def _convert_detector_result_to_mc_tensor(
        self, 
        detector_results: List[dict], 
        strong_unlabeled_images: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        detector.predict_with_uncertainty 결과를 MC tensor로 변환
        
        Args:
            detector_results: detector.predict_with_uncertainty의 결과 리스트
            strong_unlabeled_images: 입력 이미지 텐서
            
        Returns:
            MC tensor (T, B, N, C) 또는 None (변환 실패 시)
        """
        try:
            device = strong_unlabeled_images.device
            B = strong_unlabeled_images.shape[0]  # 배치 크기
            
            # detector 결과에서 MC 예측 정보 추출
            # detector 결과는 이미 필터링된 최종 결과이므로, 
            # MC tensor를 재구성하기 위해 단일 예측을 여러 번 복제
            mc_samples = []
            
            for batch_idx in range(B):
                if batch_idx < len(detector_results):
                    result = detector_results[batch_idx]
                    
                    if 'boxes' in result and len(result['boxes']) > 0:
                        boxes = result['boxes']  # (N, 4)
                        scores = result['scores']  # (N,)
                        labels = result['labels']  # (N,)
                        
                        # YOLO 형식으로 변환: (N, 5+num_classes)
                        num_classes = 80  # COCO 기본값 (config에서 가져와야 함)
                        yolo_pred = torch.zeros(len(boxes), 5 + num_classes, device=device)
                        
                        # 박스 좌표 (x, y, w, h)
                        yolo_pred[:, :4] = boxes
                        
                        # objectness score
                        yolo_pred[:, 4] = scores
                        
                        # 클래스 one-hot encoding
                        for i, label in enumerate(labels):
                            if 0 <= label < num_classes:
                                yolo_pred[i, 5 + label] = 1.0
                        
                        mc_samples.append(yolo_pred)
                    else:
                        # 빈 예측
                        mc_samples.append(torch.zeros(0, 5 + 80, device=device))
                else:
                    # 빈 예측
                    mc_samples.append(torch.zeros(0, 5 + 80, device=device))
            
            # MC tensor 생성 (단일 예측을 여러 번 복제하여 MC 효과 시뮬레이션)
            # 실제로는 detector에서 이미 MC 샘플링이 완료되었으므로, 
            # 단일 예측을 기반으로 일관성 loss 계산
            if mc_samples:
                # 가장 긴 예측 길이에 맞춰 패딩
                max_len = max(len(sample) for sample in mc_samples)
                padded_samples = []
                
                for sample in mc_samples:
                    if len(sample) < max_len:
                        # 패딩
                        padding = torch.zeros(max_len - len(sample), sample.shape[1], device=device)
                        padded_sample = torch.cat([sample, padding], dim=0)
                    else:
                        padded_sample = sample
                    padded_samples.append(padded_sample)
                
                # 배치 차원으로 스택
                batch_tensor = torch.stack(padded_samples, dim=0)  # (B, N, C)
                
                # MC 차원 추가 (단일 예측을 3번 복제하여 MC 효과 시뮬레이션)
                mc_tensor = batch_tensor.unsqueeze(0).repeat(3, 1, 1, 1)  # (3, B, N, C)
                
                return mc_tensor
            else:
                return None
                
        except Exception as e:
            print(f"Error converting detector result to MC tensor: {e}")
            return None
    
    def _compute_iou_matrix(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        두 박스 세트 간의 IoU 행렬 계산
        """
        # 간단한 IoU 계산 (xywh format)
        def box_iou(box1, box2):
            # 박스 차원 확인 및 수정
            if len(box1) != 4:
                print(f"Warning: box1 has {len(box1)} dimensions, expected 4")
                return torch.tensor(0.0, device=box1.device)
            if len(box2) != 4:
                print(f"Warning: box2 has {len(box2)} dimensions, expected 4")
                return torch.tensor(0.0, device=box2.device)
                
            # xywh를 xyxy로 변환
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            
            # xyxy 좌표
            x1_1, y1_1, x2_1, y2_1 = x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2
            x1_2, y1_2, x2_2, y2_2 = x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2
            
            # 교집합
            x1_i = torch.max(x1_1, x1_2)
            y1_i = torch.max(y1_1, y1_2)
            x2_i = torch.min(x2_1, x2_2)
            y2_i = torch.min(y2_1, y2_2)
            
            intersection = torch.clamp(x2_i - x1_i, 0) * torch.clamp(y2_i - y1_i, 0)
            
            # 합집합
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            return intersection / (union + 1e-8)
        
        # IoU 행렬 계산
        iou_matrix = torch.zeros(len(boxes1), len(boxes2), device=boxes1.device)
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                iou_matrix[i, j] = box_iou(box1, box2)
        
        return iou_matrix
    
    def _match_predictions_to_pseudo(self, iou_matrix: torch.Tensor, obj_scores: torch.Tensor) -> List[Tuple[int, int]]:
        """
        IoU와 objectness score를 기반으로 예측과 pseudo label 매칭
        """
        matches = []
        used_predictions = set()
        used_pseudo = set()
        
        # IoU 임계값
        iou_threshold = 0.5
        
        # 높은 IoU부터 매칭
        while True:
            if iou_matrix.numel() == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < iou_threshold:
                break
            
            pred_idx, pseudo_idx = (iou_matrix == max_iou).nonzero(as_tuple=True)
            if len(pred_idx) == 0:
                break
            
            pred_idx, pseudo_idx = pred_idx[0].item(), pseudo_idx[0].item()
            
            if pred_idx not in used_predictions and pseudo_idx not in used_pseudo:
                # Objectness score가 충분히 높은 경우만 매칭
                if obj_scores[pred_idx] > 0.25:
                    matches.append((pred_idx, pseudo_idx))
                    used_predictions.add(pred_idx)
                    used_pseudo.add(pseudo_idx)
            
            # 해당 위치를 0으로 설정하여 다음 반복에서 제외
            iou_matrix[pred_idx, pseudo_idx] = 0
        
        return matches
    
    def _compute_adaptive_weights(
        self, 
        pseudo_consistency_loss: torch.Tensor,
        mc_uncertainty_loss: torch.Tensor
    ) -> Tuple[float, float]:
        """
        학습 진행에 따른 적응적 가중치 계산
        """
        with torch.no_grad():
            momentum = 0.99
            self.running_pseudo_consistency = self.running_pseudo_consistency * momentum + pseudo_consistency_loss.detach() * (1-momentum)
            self.running_mc_uncertainty = self.running_mc_uncertainty * momentum + mc_uncertainty_loss.detach() * (1-momentum)
            self.num_updates += 1
        
        # 정규화된 loss magnitudes
        eps = 1e-6
        pseudo_norm = self.running_pseudo_consistency / (self.running_pseudo_consistency + eps)
        mc_norm = self.running_mc_uncertainty / (self.running_mc_uncertainty + eps)
        
        # 적응적 가중치 계산
        progress = min(self.num_updates / 1000.0, 1.0)
        
        # 초기에는 pseudo consistency 중시, 후기에는 MC uncertainty 중시
        alpha = self.alpha * (1.5 - 0.5 * progress)  # 1.5 → 1.0
        beta = self.beta * (0.5 + 0.5 * progress)    # 0.5 → 1.0
        
        return float(alpha), float(beta)


def create_mc_consistency_loss(
    alpha: float = 1.0,
    beta: float = 0.5,
    temperature: float = 1.0,
    uncertainty_threshold: float = 0.1,
    use_adaptive_weighting: bool = True
) -> MCDropoutConsistencyLoss:
    """
    MC Dropout Consistency Loss 팩토리 함수
    """
    return MCDropoutConsistencyLoss(
        alpha=alpha,
        beta=beta,
        temperature=temperature,
        uncertainty_threshold=uncertainty_threshold,
        use_adaptive_weighting=use_adaptive_weighting
    ) 