import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List, Union


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
        # 🔧 YOLOv8 feature maps 처리
        if isinstance(predictions, (list, tuple)):
            # Raw feature maps인 경우 - YOLOv8 training mode output
            if len(predictions) > 0 and len(predictions[0].shape) == 4:  # [B, C, H, W]
                # Feature maps를 detection format으로 변환
                predictions = convert_yolov8_featuremaps_to_detections(
                    predictions, 
                    img_size=640,  # TODO: config에서 가져오기
                    num_classes=80  # TODO: 동적으로 설정
                )
            else:
                # 이미 processed된 predictions list인 경우 첫 번째만 사용
                predictions = predictions[0] if len(predictions) > 0 else torch.empty(0)
        
        # 예측이 비어있거나 None인 경우 처리
        if predictions is None or (isinstance(predictions, torch.Tensor) and predictions.numel() == 0):
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
        
        if num_predictions == 0:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return zero_loss, zero_loss, zero_loss
        
        # 예측 차원 분해: [x, y, w, h, objectness, class_logits...]
        num_classes = predictions.shape[-1] - 5
        if num_classes <= 0:
            raise ValueError(f"Invalid prediction dimensions: {predictions.shape}. Expected at least 5 + num_classes.")
        
        # 타겟이 없는 경우 처리
        if targets.numel() == 0:
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
        
        # Positive 샘플 할당 (간단한 버전)
        # TODO: 더 정교한 target assignment 구현 필요
        pos_mask = torch.zeros(batch_size, num_predictions, dtype=torch.bool, device=device)
        
        if len(targets) > 0:
            # 각 타겟에 대해 가장 가까운 예측 찾기 (단순 버전)
            for i in range(len(targets)):
                batch_idx = target_batch_idx[i]
                if 0 <= batch_idx < batch_size:
                    # IoU 기반 할당 또는 center distance 기반 할당
                    target_box = target_boxes[i:i+1]  # [1, 4]
                    batch_pred_boxes = pred_boxes[batch_idx]  # [num_pred, 4]
                    
                    # Center distance 계산 (간단한 방법)
                    pred_centers = batch_pred_boxes[:, :2]  # [num_pred, 2]
                    target_center = target_box[:, :2]       # [1, 2]
                    distances = torch.sum((pred_centers - target_center) ** 2, dim=1)  # [num_pred]
                    
                    # 가장 가까운 예측을 positive로 할당
                    closest_idx = torch.argmin(distances)
                    pos_mask[batch_idx, closest_idx] = True
        
        # Box Loss 계산 (positive 샘플에 대해서만)
        box_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if pos_mask.any():
            # positive 예측과 해당 타겟 매칭
            pos_pred_boxes = pred_boxes[pos_mask]  # [num_pos, 4]
            
            # 해당하는 타겟 박스 찾기
            pos_indices = torch.where(pos_mask)
            pos_target_boxes = []
            
            for batch_idx, pred_idx in zip(pos_indices[0], pos_indices[1]):
                # 해당 배치의 타겟 중에서 매칭되는 것 찾기
                batch_targets = targets[target_batch_idx == batch_idx]
                if len(batch_targets) > 0:
                    # 첫 번째 타겟 사용 (더 정교한 매칭 필요)
                    pos_target_boxes.append(batch_targets[0, 2:6])
            
            if pos_target_boxes:
                pos_target_boxes = torch.stack(pos_target_boxes)  # [num_pos, 4]
                
                if self.bbox_loss_type == 'giou':
                    box_loss = bbox_giou(pos_pred_boxes, pos_target_boxes, xywh=True).mean()
                elif self.bbox_loss_type == 'ciou':
                    box_loss = bbox_ciou(pos_pred_boxes, pos_target_boxes, xywh=True).mean()
                else:
                    box_loss = F.mse_loss(pos_pred_boxes, pos_target_boxes)
                
                # 안정성을 위해 loss 값을 양수로 클리핑
                box_loss = torch.clamp(box_loss, min=0.0)
                box_loss = box_loss * self.box_loss_gain
        
        # Classification Loss 계산 (positive 샘플에 대해서만)
        cls_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if pos_mask.any():
            pos_pred_cls = pred_cls[pos_mask]  # [num_pos, num_classes]
            
            # 해당하는 타겟 클래스 찾기
            pos_target_cls = []
            pos_indices = torch.where(pos_mask)
            
            for batch_idx, pred_idx in zip(pos_indices[0], pos_indices[1]):
                batch_targets = targets[target_batch_idx == batch_idx]
                if len(batch_targets) > 0:
                    pos_target_cls.append(batch_targets[0, 1].long())
            
            if pos_target_cls:
                pos_target_cls = torch.stack(pos_target_cls)  # [num_pos]
                
                if self.focal_loss_gamma > 0:
                    cls_loss = focal_loss(pos_pred_cls, pos_target_cls, gamma=self.focal_loss_gamma)
                else:
                    cls_loss = F.cross_entropy(pos_pred_cls, pos_target_cls, label_smoothing=self.label_smoothing)
                
                cls_loss = cls_loss * self.cls_loss_gain
        
        # Objectness Loss 계산 (모든 샘플에 대해)
        obj_targets = pos_mask.float()  # positive는 1, negative는 0
        obj_loss = F.binary_cross_entropy_with_logits(
            pred_obj, obj_targets, reduction='mean'
        ) * self.obj_loss_gain
        
        # Uncertainty weighting 적용
        if uncertainty_weight is not None:
            # Uncertainty가 높을수록 낮은 가중치 적용
            weight = 1.0 / (1.0 + uncertainty_weight)
            box_loss = box_loss * weight.mean()
            cls_loss = cls_loss * weight.mean()
            obj_loss = obj_loss * weight.mean()
        
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
        Calculate semi-supervised YOLO losses with uncertainty weighting.
        
        Args:
            predictions: Model predictions
            targets: Pseudo labels from teacher model
            prediction_variance: Variance from MC Dropout predictions
        
        Returns:
            Tuple of (box_loss, cls_loss, obj_loss)
        """
        uncertainty_weight = None
        if prediction_variance is not None:
            uncertainty_weight = self.compute_uncertainty_weight(prediction_variance)
        
        return super().forward(predictions, targets, uncertainty_weight=uncertainty_weight)


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
    batch_size = feature_maps[0].shape[0]
    device = feature_maps[0].device
    all_predictions = []
    
    # YOLOv8 anchor strides
    strides = [8, 16, 32]  # P3, P4, P5
    
    for i, (feat, stride) in enumerate(zip(feature_maps, strides)):
        b, c, h, w = feat.shape
        
        # YOLOv8 채널 분해: 64 (bbox DFL) + 80 (classes) = 144
        bbox_channels = 64  # 4 coordinates * 16 DFL bins
        class_channels = num_classes  # 80
        
        # 채널 분리
        bbox_feat = feat[:, :bbox_channels, :, :]  # [b, 64, h, w] - DFL regression
        cls_feat = feat[:, bbox_channels:bbox_channels+class_channels, :, :]  # [b, 80, h, w] - classes
        
        # Grid 생성 (YOLOv8 방식)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Feature map을 [b, h, w, channels] 형태로 변환
        bbox_pred = bbox_feat.permute(0, 2, 3, 1).contiguous()  # [b, h, w, 64]
        cls_pred = cls_feat.permute(0, 2, 3, 1).contiguous()    # [b, h, w, 80]
        
        # DFL regression을 좌표로 변환
        bbox_pred = bbox_pred.view(b, h, w, 4, 16)  # [b, h, w, 4, 16]
        bbox_pred = F.softmax(bbox_pred, dim=-1)    # DFL softmax
        
        # DFL 가중합으로 실제 좌표 계산
        dfl_range = torch.arange(16, device=device, dtype=torch.float32)
        bbox_pred = (bbox_pred * dfl_range.view(1, 1, 1, 1, 16)).sum(dim=-1)  # [b, h, w, 4]
        
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
        
        # Objectness는 클래스 최대값으로 근사 (YOLOv8는 별도 objectness 없음)
        objectness = torch.sigmoid(cls_pred.max(dim=-1, keepdim=True)[0])  # [b, h, w, 1]
        
        # Flatten to [b, h*w, features]
        xywh_flat = xywh.view(b, h*w, 4)
        obj_flat = objectness.view(b, h*w, 1)
        cls_flat = cls_pred.view(b, h*w, num_classes)
        
        # Final prediction: [x, y, w, h, obj, cls1, cls2, ...]
        prediction = torch.cat([xywh_flat, obj_flat, cls_flat], dim=-1)  # [b, h*w, 5+num_classes]
        all_predictions.append(prediction)
    
    # 모든 스케일 결합
    final_predictions = torch.cat(all_predictions, dim=1)  # [b, total_anchors, 5+num_classes]
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
    try:
        # DDP 지원
        if hasattr(model, 'module'):
            student_model = model.module.student_model
        else:
            student_model = model.student_model
        
        student_model.model.train()
        
        # Student 모델로 Strong Augmentation된 unlabeled 데이터 예측
        student_predictions = student_model.model(strong_unlabeled_images)
        
        # Pseudo Label을 YOLO target 형식으로 변환
        pseudo_targets = []
        for i, pseudo_label in enumerate(high_quality_pseudo_labels[:len(strong_unlabeled_images)]):
            if 'boxes' in pseudo_label and len(pseudo_label['boxes']) > 0:
                boxes = pseudo_label['boxes']
                batch_labels = torch.zeros((len(boxes), 6), device=device)
                batch_labels[:, 0] = i  # batch index
                batch_labels[:, 1:] = boxes  # [class_id, x, y, w, h]
                pseudo_targets.append(batch_labels)
        
        if pseudo_targets:
            pseudo_targets = torch.cat(pseudo_targets, dim=0)
            
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
                        reliability_score = pl['uncertainty_stats'].get('reliability_score', 0.8)
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
                    lbox, lcls, lobj = semi_loss_fn(pred, pseudo_targets, prediction_variance)
                    unlabeled_data_loss = (lbox + lcls + lobj) * unlabeled_weight
                    return unlabeled_data_loss
                
            except Exception as e:
                logger.debug(f"Semi-Supervised YOLO Loss calculation failed: {e}")
        
        # 실패 시 기본값 반환
        return torch.tensor(0.0, device=device, requires_grad=True)
        
    except Exception as e:
        logger.debug(f"Unlabeled Data Loss calculation failed: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True) 