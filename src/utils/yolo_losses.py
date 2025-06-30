import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


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
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        uncertainty_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate YOLO losses.
        
        Args:
            predictions: Model predictions (B, N, 5+C) where N is number of anchors
                        Format: [x, y, w, h, obj_conf, cls1, cls2, ...]
            targets: Ground truth (M, 6) where M is number of objects
                    Format: [batch_idx, class_id, x_center, y_center, width, height]
            uncertainty_weight: Optional uncertainty weighting tensor
        
        Returns:
            Tuple of (box_loss, cls_loss, obj_loss)
        """
        device = predictions.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device) 
        lobj = torch.zeros(1, device=device)
        
        # Handle empty targets
        if targets.shape[0] == 0:
            return lbox, lcls, lobj
        
        # Separate predictions
        pred_boxes = predictions[..., :4]  # (B, N, 4)
        pred_obj = predictions[..., 4]     # (B, N)
        pred_cls = predictions[..., 5:]    # (B, N, C)
        
        batch_size, num_anchors, num_classes = pred_cls.shape
        
        # Initialize objectness targets
        obj_targets = torch.zeros_like(pred_obj)
        
        # 안전성 체크: targets에서 batch_idx와 class_id 범위 확인
        if targets.shape[0] > 0:
            max_batch_idx = targets[:, 0].max().item()
            max_class_id = targets[:, 1].max().item()
            
            # batch_idx 범위 체크
            if max_batch_idx >= batch_size:
                print(f"Warning: max_batch_idx ({max_batch_idx}) >= batch_size ({batch_size})")
                # 유효한 batch_idx만 필터링
                valid_mask = targets[:, 0] < batch_size
                targets = targets[valid_mask]
                
                if targets.shape[0] == 0:
                    return lbox, lcls, lobj
            
            # class_id 범위 체크 (더 엄격하게)
            if max_class_id >= num_classes:
                print(f"Warning: max_class_id ({max_class_id}) >= num_classes ({num_classes})")
                # 유효한 class_id만 필터링 (0 <= class_id < num_classes)
                valid_mask = (targets[:, 1] >= 0) & (targets[:, 1] < num_classes)
                targets = targets[valid_mask]
                
                if targets.shape[0] == 0:
                    return lbox, lcls, lobj
            
            # 추가 안전성: 음수 클래스 ID 체크
            min_class_id = targets[:, 1].min().item()
            if min_class_id < 0:
                print(f"Warning: negative class_id ({min_class_id}) found")
                valid_mask = targets[:, 1] >= 0
                targets = targets[valid_mask]
                
                if targets.shape[0] == 0:
                    return lbox, lcls, lobj
        
        # Process targets for each batch
        for batch_idx in range(batch_size):
            # Get targets for this batch
            batch_targets = targets[targets[:, 0] == batch_idx]
            if len(batch_targets) == 0:
                continue
            
            # Extract target information
            target_cls = batch_targets[:, 1].long()  # Class indices
            target_boxes = batch_targets[:, 2:6]     # Box coordinates (xywh format)
            
            # For simplicity, assign each target to the closest anchor
            # In practice, YOLO uses more sophisticated anchor assignment
            num_targets = len(batch_targets)
            
            # Simple assignment: match each target to one anchor
            for i, (cls_id, box) in enumerate(zip(target_cls, target_boxes)):
                # Find best anchor (simplified - normally based on IoU)
                anchor_idx = i % num_anchors  # Simple round-robin assignment
                
                # 추가 안전성 체크
                if anchor_idx >= num_anchors:
                    continue
                if cls_id >= num_classes:
                    continue
                if batch_idx >= batch_size:
                    continue
                
                # Set objectness target
                obj_targets[batch_idx, anchor_idx] = 1.0
                
                # Calculate box loss
                try:
                    pred_box = pred_boxes[batch_idx, anchor_idx]
                    if self.bbox_loss_type == 'giou':
                        box_loss = bbox_giou(pred_box.unsqueeze(0), box.unsqueeze(0), xywh=True)
                    else:  # ciou
                        box_loss = bbox_ciou(pred_box.unsqueeze(0), box.unsqueeze(0), xywh=True)
                    
                    if torch.isfinite(box_loss):
                        lbox += box_loss.mean()
                except Exception as e:
                    print(f"Box loss calculation error: {e}")
                    continue
                
                # Calculate classification loss
                try:
                    pred_cls_single = pred_cls[batch_idx, anchor_idx].unsqueeze(0)
                    
                    # 안전한 클래스 ID 처리
                    safe_cls_id = torch.clamp(cls_id, 0, num_classes - 1).long()
                    target_cls_single = safe_cls_id.unsqueeze(0)
                    
                    # 추가 안전성 체크
                    if target_cls_single.max() >= num_classes or target_cls_single.min() < 0:
                        print(f"Invalid class ID after clamp: {target_cls_single.item()}, skipping")
                        continue
                    
                    if self.focal_loss_gamma > 0:
                        cls_loss = focal_loss(pred_cls_single, target_cls_single, gamma=self.focal_loss_gamma)
                    else:
                        cls_loss = F.cross_entropy(
                            pred_cls_single, 
                            target_cls_single, 
                            label_smoothing=self.label_smoothing
                        )
                    
                    if torch.isfinite(cls_loss):
                        lcls += cls_loss
                except Exception as e:
                    print(f"Classification loss calculation error: {e}")
                    continue
        
        # Objectness loss (binary cross entropy)
        try:
            lobj = F.binary_cross_entropy_with_logits(pred_obj, obj_targets, reduction='mean')
        except Exception as e:
            print(f"Objectness loss calculation error: {e}")
            lobj = torch.zeros(1, device=device)
        
        # Apply uncertainty weighting if provided
        if uncertainty_weight is not None:
            # uncertainty_weight shape should match the loss shapes
            weight = uncertainty_weight.mean() if uncertainty_weight.numel() > 1 else uncertainty_weight
            if torch.isfinite(weight):
                lbox *= weight
                lcls *= weight
                lobj *= weight
        
        # Apply loss gains
        lbox *= self.box_loss_gain
        lcls *= self.cls_loss_gain
        lobj *= self.obj_loss_gain
        
        # NaN/Inf 체크 및 처리
        if not torch.isfinite(lbox):
            lbox = torch.zeros(1, device=device)
        if not torch.isfinite(lcls):
            lcls = torch.zeros(1, device=device)
        if not torch.isfinite(lobj):
            lobj = torch.zeros(1, device=device)
        
        return lbox, lcls, lobj


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
    Factory function to create YOLO loss.
    
    Args:
        loss_type: 'supervised' or 'semi_supervised'
        bbox_loss_type: 'giou' or 'ciou'
        box_gain: Box loss weight
        cls_gain: Classification loss weight 
        obj_gain: Objectness loss weight
        **kwargs: Additional arguments
    
    Returns:
        YOLO loss module
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