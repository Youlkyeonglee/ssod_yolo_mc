from typing import List, Dict, Any, Union, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import logging
import time
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_validation import validate_box_coordinates, filter_invalid_predictions

class MCDropoutDetector(nn.Module):
    """Monte Carlo Dropout을 사용한 객체 탐지 불확실성 추정"""
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 10,
        dropout_rate: float = 0.1,
        box_std_threshold: float = 0.1,
        entropy_threshold: float = 0.5,
        conf_threshold: float = 0.25,
        max_pseudo_labels: int = 200,
        save_dir: Optional[Path] = None
    ):
        """
        Args:
            model: 기본 객체 탐지 모델
            num_samples: MC Dropout 샘플링 횟수
            dropout_rate: Dropout 비율
            box_std_threshold: 박스 좌표 표준편차 임계값 (초기값)
            entropy_threshold: 클래스 엔트로피 임계값 (초기값)
            conf_threshold: 신뢰도 임계값
            save_dir: 결과 저장 디렉토리
        """
        super().__init__()
        
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        self.box_std_threshold = box_std_threshold
        self.entropy_threshold = entropy_threshold
        self.conf_threshold = conf_threshold
        self.max_pseudo_labels = max_pseudo_labels
        
        # 결과 저장 디렉토리 설정
        if save_dir is None:
            save_dir = Path("runs/train/default/mcdropout")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 데이터 수집을 위한 변수들
        self.prediction_history = []
        self.uncertainty_history = []
        
        self._enable_dropout()
    
    def _enable_dropout(self):
        """모델의 모든 Dropout 레이어를 추론 시에도 활성화"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Dropout을 활성화 상태로 유지

    def get_current_thresholds(self) -> Dict[str, float]:
        """현재 동적 임계값들을 반환"""
        return {
                'box_std_threshold': self.box_std_threshold,
            'entropy_threshold': self.entropy_threshold
        }

    def reset_adaptive_thresholds(self, box_std_threshold: float = None, entropy_threshold: float = None):
        """동적 임계값을 초기값으로 리셋"""
        if box_std_threshold is not None:
            self.box_std_threshold = box_std_threshold
        
        if entropy_threshold is not None:
            self.entropy_threshold = entropy_threshold
        
        print(f"🔄 동적 임계값 리셋: box_std={self.box_std_threshold:.4f}, entropy={self.entropy_threshold:.4f}")

    @torch.no_grad()
    def predict_with_uncertainty_legacy(
        self,
        image: Union[str, torch.Tensor],
        device: Optional[str] = None,
        save_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        MC Dropout을 사용하여 불확실성을 포함한 예측 수행 (NMS 기반 클러스터링 방식)
        
        Args:
            image: 입력 이미지 (경로 또는 텐서)
            device: 실행 디바이스
            save_predictions: 예측 결과를 히스토리에 저장할지 여부
        
        Returns:
            예측 결과와 불확실성 측정값을 포함한 딕셔너리
        """
        # 모델이 있는 디바이스 자동 감지
        try:
            if hasattr(self.model, 'module'):
                model_device = next(self.model.module.parameters()).device
            else:
                model_device = next(self.model.parameters()).device
            
            if device is None:
                device = model_device
            elif isinstance(device, str) and device != str(model_device):
                print(f"⚠️  Device mismatch in legacy: model on {model_device}, requested {device}")
                device = model_device
        except Exception as e:
            print(f"❌ Device setup error in legacy: {e}")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 이미지를 텐서로 변환
        if isinstance(image, str):
            pass
        elif isinstance(image, torch.Tensor):
            try:
                image = image.to(device)
            except Exception as e:
                print(f"❌ Image device transfer error: {e}")
                device = image.device
        
        # MC Dropout 샘플링 수행
        nms_results = []
        self.model.train()  # MC Dropout 활성화
        
        try:
            for sample_idx in range(self.num_samples):
                # 모델 예측
                try:
                    if hasattr(self.model, 'module'):
                        model_device = next(self.model.module.parameters()).device
                    else:
                        model_device = next(self.model.parameters()).device
                    
                    if image.device != model_device:
                        print(f"🔄 Moving image from {image.device} to model device {model_device}")
                        image = image.to(model_device)
                    
                    pred_output = self.model(image)
                    
                except RuntimeError as device_error:
                    if "device" in str(device_error).lower():
                        print(f"🚨 Device error in MC Dropout: {device_error}")
                        try:
                            print(f"🔄 Moving model to image device: {image.device}")
                            self.model = self.model.to(image.device)
                            pred_output = self.model(image)
                        except Exception as move_error:
                            print(f"❌ Failed to move model: {move_error}")
                            raise device_error
                    else:
                        raise device_error
                
                # 예측 결과 추출
                if isinstance(pred_output, dict):
                    pred = pred_output['predictions']
                else:
                    pred = pred_output
                
                # Feature map 결합 및 NMS 적용
                processed_pred = self._process_prediction(pred)
                nms_boxes = self._apply_nms_to_sample(processed_pred, sample_idx)
                nms_results.append(nms_boxes)
            
            # NMS 결과들 간의 클러스터링 및 불확실성 계산
            final_results = self._compute_uncertainty_from_nms_results(nms_results)
            
            # 예측 히스토리에 저장
            if save_predictions and len(self.uncertainty_history) < 10:
                self.uncertainty_history.append(final_results)
            
            return final_results
            
        except Exception as e:
            print(f"Error in MC Dropout prediction: {str(e)}")
            return None
    
    def _process_prediction(self, pred):
        """예측 결과 처리 및 feature map 결합"""
        if isinstance(pred, (list, tuple)):
            batch_pred = []
            expected_classes = None

            for i, p in enumerate(pred):
                current_classes = p.shape[-1] - 5
                
                if expected_classes is None:
                    expected_classes = current_classes
                
                if current_classes != expected_classes:
                    print(f"Warning: Feature map {i} has {current_classes} classes, expected {expected_classes}. Skipping.")
                    continue
    
                reshaped = p.view(p.shape[0], -1, p.shape[-1])
                batch_pred.append(reshaped)
            
            if batch_pred:
                pred = torch.cat(batch_pred, dim=1)
            else:
                if expected_classes is not None:
                    pred = torch.zeros(p.shape[0], 0, 5 + expected_classes).to(p.device)
                else:
                    pred = torch.zeros(1, 0, 85).to(p.device)
    
        return pred
    
    def _apply_nms_to_sample(self, pred, sample_idx):
        """각 MC 샘플에 NMS 적용"""
        batch_size = pred.shape[0]
        nms_results = []
        
        for batch_idx in range(batch_size):
                # 신뢰도 기반 필터링
            conf_scores = pred[batch_idx, :, 4]
            confident_mask = conf_scores > self.conf_threshold
                
            if not confident_mask.any():
                nms_results.append({
                    'boxes': torch.empty(0, 4, device=pred.device),
                    'scores': torch.empty(0, device=pred.device),
                    'labels': torch.empty(0, dtype=torch.long, device=pred.device)
                })
                continue
            
            # 필터링된 예측
            filtered_pred = pred[batch_idx, confident_mask]
            boxes = filtered_pred[:, :4]
            scores = filtered_pred[:, 4]
            class_scores = filtered_pred[:, 5:]
            
            # 클래스 예측
            predicted_classes = torch.argmax(class_scores, dim=-1)
            
            # 박스 좌표 검증
            if boxes.numel() > 0:
                boxes_np = boxes.cpu().numpy()
                validated_boxes, valid_mask = validate_box_coordinates(boxes_np, (640, 640))
                
                if np.any(valid_mask):
                    valid_boxes = torch.from_numpy(validated_boxes[valid_mask]).to(boxes.device)
                    valid_scores = scores[valid_mask]
                    valid_classes = predicted_classes[valid_mask]
                else:
                    valid_boxes = torch.empty(0, 4, device=boxes.device)
                    valid_scores = torch.empty(0, device=scores.device)
                    valid_classes = torch.empty(0, dtype=torch.long, device=predicted_classes.device)
            else:
                valid_boxes = torch.empty(0, 4, device=boxes.device)
                valid_scores = torch.empty(0, device=scores.device)
                valid_classes = torch.empty(0, dtype=torch.long, device=predicted_classes.device)
            
            nms_results.append({
                'boxes': valid_boxes,
                'scores': valid_scores,
                'labels': valid_classes
            })
        
        return nms_results
    
    def _compute_uncertainty_from_nms_results(self, nms_results):
        """NMS 결과들 간의 클러스터링 및 불확실성 계산"""
        batch_size = len(nms_results[0])
        final_results = []
        
        for batch_idx in range(batch_size):
            # 모든 MC 샘플에서 해당 배치의 박스들 수집
            all_boxes = []
            all_scores = []
            all_labels = []
            
            for sample_idx, sample_result in enumerate(nms_results):
                boxes = sample_result[batch_idx]['boxes']
                scores = sample_result[batch_idx]['scores']
                labels = sample_result[batch_idx]['labels']
                
                if len(boxes) > 0:
                    all_boxes.extend(boxes.cpu().numpy())
                    all_scores.extend(scores.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            if len(all_boxes) == 0:
                final_results.append({
                    'boxes': torch.empty(0, 4),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0, dtype=torch.long),
                    'box_std': torch.empty(0, 4),
                    'class_entropy': torch.empty(0)
                })
                continue
            
            # 박스 클러스터링
            clusters = self._cluster_similar_boxes(all_boxes, all_scores, all_labels)
            
            # 각 클러스터에서 불확실성 계산
            final_boxes = []
            final_scores = []
            final_labels = []
            final_box_stds = []
            final_entropies = []
            
            for cluster in clusters:
                if len(cluster['boxes']) >= 2:  # 최소 2개 이상의 박스가 있어야 불확실성 계산 가능
                    # 클러스터 내 박스들의 평균과 표준편차 계산
                    cluster_boxes = torch.stack(cluster['boxes'])
                    cluster_scores = torch.stack(cluster['scores'])
                    cluster_labels = torch.stack(cluster['labels'])
                    
                    mean_box = torch.mean(cluster_boxes, dim=0)
                    box_std = torch.std(cluster_boxes, dim=0)
                    
                    # 불확실성 계산 (상대적 표준편차)
                    relative_std = box_std / (mean_box + 1e-8)
                    uncertainty = relative_std.mean().item()
                    
                    # 클러스터 크기에 따른 신뢰도 조정
                    support_ratio = len(cluster['boxes']) / self.num_samples
                    
                    # 불확실성 임계값 체크 (클러스터 크기도 고려)
                    if uncertainty < self.box_std_threshold and support_ratio > 0.3:  # 최소 30% 샘플에서 검출
                        final_boxes.append(mean_box)
                        final_scores.append(torch.mean(cluster_scores))
                        final_labels.append(torch.mode(cluster_labels)[0])  # 가장 빈번한 클래스
                        final_box_stds.append(box_std)
                        final_entropies.append(torch.tensor(uncertainty))
            
            if final_boxes:
                    result = {
                    'boxes': torch.stack(final_boxes),
                    'scores': torch.stack(final_scores),
                    'labels': torch.stack(final_labels),
                    'box_std': torch.stack(final_box_stds),
                    'class_entropy': torch.stack(final_entropies)
                }
            else:
                result = {
                    'boxes': torch.empty(0, 4),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0, dtype=torch.long),
                    'box_std': torch.empty(0, 4),
                    'class_entropy': torch.empty(0)
                }
            
            final_results.append(result)
        
        return final_results
    
    def _cluster_similar_boxes(self, boxes, scores, labels, iou_threshold=0.5):
        """IoU 기반으로 유사한 박스들을 클러스터링"""
        if len(boxes) == 0:
            return []
        
        # 박스를 numpy 배열로 변환
        boxes = np.array(boxes)
        scores = np.array(scores)
        labels = np.array(labels)
        
        clusters = []
        used = [False] * len(boxes)
        
        for i in range(len(boxes)):
            if used[i]:
                continue
            
            # 새로운 클러스터 시작
            cluster = {
                'boxes': [torch.from_numpy(boxes[i]).float()],
                'scores': [torch.tensor(scores[i]).float()],
                'labels': [torch.tensor(labels[i]).long()]
            }
            used[i] = True
            
            # 유사한 박스들 찾기
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                
                # 같은 클래스이고 IoU가 임계값 이상인 경우
                if labels[i] == labels[j]:
                    iou = self._compute_iou(boxes[i], boxes[j])
                    if iou >= iou_threshold:
                        cluster['boxes'].append(torch.from_numpy(boxes[j]).float())
                        cluster['scores'].append(torch.tensor(scores[j]).float())
                        cluster['labels'].append(torch.tensor(labels[j]).long())
                        used[j] = True
            
            clusters.append(cluster)
        
        return clusters
    
    def _compute_iou(self, box1, box2):
        """두 박스 간의 IoU 계산"""
        # YOLO 형식 (x_center, y_center, width, height)를 (x1, y1, x2, y2)로 변환
        x1_1, y1_1, w1, h1 = box1
        x1_2, y1_2, w2, h2 = box2
        
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # 교집합 계산
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 합집합 계산
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_mean_boxes(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """여러 예측의 평균 박스 계산
        
        Args:
            predictions: 여러 번의 박스 예측 결과 리스트
        
        Returns:
            평균 박스 좌표
        """
        if not predictions:
            return torch.zeros((0, 4))
        
        # 모든 예측을 스택으로 쌓기
        stacked_preds = torch.stack(predictions)
        
        # 평균 계산
        mean_boxes = torch.mean(stacked_preds, dim=0)
        
        return mean_boxes
    
    def _compute_box_std(
        self,
        predictions: List[torch.Tensor],
        mean_boxes: torch.Tensor
    ) -> torch.Tensor:
        """박스 좌표의 표준편차 계산
        
        Args:
            predictions: 여러 번의 박스 예측 결과 리스트
            mean_boxes: 평균 박스 좌표
        
        Returns:
            박스 좌표의 표준편차
        """
        if not predictions:
            return torch.zeros((0, 4))
        
        # 모든 예측을 스택으로 쌓기
        stacked_preds = torch.stack(predictions)
        
        # 표준편차 계산
        box_std = torch.std(stacked_preds, dim=0)
        
        return box_std
    
    def _compute_entropy_regularization(self, mc_tensor: torch.Tensor) -> torch.Tensor:
        """
        Entropy Regularization 계산
        
        각 MC 샘플의 예측 엔트로피를 정규화하여 confident predictions 유도
        """
        try:
            T, B, N, C = mc_tensor.shape
            
            # Softmax를 class probabilities에만 적용
            cls_probs = F.softmax(mc_tensor[..., 5:], dim=-1)  # (T, B, N, num_classes)
            
            # 각 MC 샘플의 class entropy 계산
            eps = 1e-8
            cls_entropy = -(cls_probs * torch.log(cls_probs + eps)).sum(dim=-1)  # (T, B, N)
            
            # MC 샘플들의 평균 엔트로피
            mean_entropy = cls_entropy.mean(dim=0)  # (B, N)
            
            # Objectness confidence도 고려
            obj_conf = torch.sigmoid(mc_tensor[..., 4])  # (T, B, N)
            obj_entropy = -(obj_conf * torch.log(obj_conf + eps) + 
                           (1 - obj_conf) * torch.log(1 - obj_conf + eps)).mean(dim=0)
            
            total_entropy = mean_entropy + 0.5 * obj_entropy  # (B, N)
            
            # NaN/Inf 체크
            if torch.isnan(total_entropy).any() or torch.isinf(total_entropy).any():
                print(f"⚠️  Invalid entropy loss detected: {total_entropy}")
                total_entropy = torch.zeros_like(total_entropy)
            
            return total_entropy  # (B, N)
            
        except Exception as e:
            print(f"⚠️  Entropy regularization calculation failed: {e}")
            return torch.zeros(mc_tensor.shape[1], mc_tensor.shape[2], device=mc_tensor.device)
    
    def filter_predictions(
        self,
        predictions: Dict[str, Any],
        box_std_threshold: float = None,
        entropy_threshold: float = None
    ) -> Dict[str, Any]:
        """불확실성이 높은 예측 필터링
        
        Args:
            predictions: predict_with_uncertainty의 결과
            box_std_threshold: 박스 표준편차 임계값
            entropy_threshold: 엔트로피 임계값
        
        Returns:
            필터링된 예측 결과
        """
        if box_std_threshold is None:
            box_std_threshold = self.box_std_threshold
        if entropy_threshold is None:
            entropy_threshold = self.entropy_threshold
        
        # 불확실성 측정값 추출
        box_std = predictions['box_std']
        class_entropy = predictions['class_entropy']
        mean_boxes = predictions['mean_boxes']
        
        # 불확실성이 낮은 예측만 선택
        confident_mask = (box_std < box_std_threshold) & (class_entropy < entropy_threshold)
        
        filtered_predictions = {
            'boxes': mean_boxes[confident_mask],
            'box_std': box_std[confident_mask],
            'class_entropy': class_entropy[confident_mask]
        }
        
        return filtered_predictions
    
    def predict_with_uncertainty(
        self,
        image: Union[str, torch.Tensor],
        device: Optional[str] = None,
        save_predictions: bool = True,
        reliability_threshold: Optional[float] = None,
        config: Optional[dict] = None
    ) -> Dict[str, Any]:
        """단순화된 불확실성 예측 (기존 legacy 함수 사용)
        
        Args:
            image: 입력 이미지
            device: 디바이스
            save_predictions: 예측 저장 여부
            reliability_threshold: 사용하지 않음 (legacy 함수에서 처리)
            config: 사용하지 않음 (legacy 함수에서 처리)
        
        Returns:
            box_std_threshold와 entropy_threshold로 필터링된 예측 결과
        """
        # 기존 predict_with_uncertainty_legacy 호출 (이미 충분한 필터링 제공)
        result = self.predict_with_uncertainty_legacy(image, device, save_predictions)
        
        if result is None:
            print("❌ predict_with_uncertainty_legacy가 None을 반환했습니다")
            return None
        
        # 결과에 추가 정보 추가 (기존 구조 유지)
        enhanced_results = []
        for batch_result in result:
            enhanced_result = {
                'boxes': batch_result['boxes'],
                'scores': batch_result['scores'],
                'labels': batch_result['labels'],
                'box_std': batch_result['box_std'],
                'class_entropy': batch_result['class_entropy'],
                'reliability_stats': {
                    'total_detections': len(batch_result['boxes']),
                    'avg_box_std': batch_result['box_std'].mean().item() if len(batch_result['box_std']) > 0 else 0.0,
                    'avg_class_entropy': batch_result['class_entropy'].mean().item() if len(batch_result['class_entropy']) > 0 else 0.0,
                    'filtered_by_uncertainty': True
                }
            }
            enhanced_results.append(enhanced_result)
        
        return enhanced_results

class MCLoss(nn.Module):
    """
    Monte Carlo Dropout 기반 불확실성 손실 함수
    
    MC Loss = α * Epistemic_Loss + β * Predictive_Variance_Loss + γ * Entropy_Regularization
    """
    
    def __init__(
        self,
        alpha: float = 1.0,          # Epistemic uncertainty weight
        beta: float = 0.5,           # Predictive variance weight  
        gamma: float = 0.3,          # Entropy regularization weight
        temperature: float = 1.0,    # Temperature scaling for uncertainty
        adaptive_weighting: bool = True  # 적응적 가중치 조정
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.adaptive_weighting = adaptive_weighting
        
        # 적응적 가중치를 위한 running statistics
        self.register_buffer('running_epistemic_mean', torch.tensor(0.0))
        self.register_buffer('running_variance_mean', torch.tensor(0.0))
        self.register_buffer('running_entropy_mean', torch.tensor(0.0))
        self.register_buffer('num_updates', torch.tensor(0))
        
    def forward(
        self,
        mc_predictions: List[torch.Tensor],  # MC sampling 결과들
        ground_truth: Optional[torch.Tensor] = None,  # GT (supervised에서만)
        reduction: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        MC Loss 계산
        
        Args:
            mc_predictions: MC Dropout 샘플링 결과 리스트 [T x (B, N, C)]
            ground_truth: Ground truth (supervised learning에서만 제공)
            reduction: 'mean', 'sum', 'none'
            
        Returns:
            Dict containing loss components and total MC loss
        """
        if len(mc_predictions) < 2:
            raise ValueError("MC Loss requires at least 2 MC samples")
            
        # MC predictions를 텐서로 변환: (T, B, N, C)
        mc_tensor = torch.stack(mc_predictions, dim=0)  # (T, B, N, C)
        T, B, N, C = mc_tensor.shape
        
        # 1. Epistemic Uncertainty Loss (모델 불확실성)
        epistemic_loss = self._compute_epistemic_loss(mc_tensor)
        
        # 2. Predictive Variance Loss (예측 분산)
        variance_loss = self._compute_predictive_variance_loss(mc_tensor)
        
        # 3. Entropy Regularization (예측 분포 정규화)
        entropy_loss = self._compute_entropy_regularization(mc_tensor)
        
        # 4. 적응적 가중치 조정 (선택적)
        if self.adaptive_weighting:
            adaptive_weights = self._compute_adaptive_weights(
                epistemic_loss, variance_loss, entropy_loss
            )
            alpha, beta, gamma = adaptive_weights
        else:
            alpha, beta, gamma = self.alpha, self.beta, self.gamma
            
        # 5. Total MC Loss 계산
        mc_loss = alpha * epistemic_loss + beta * variance_loss + gamma * entropy_loss
        
        # 6. Reduction 적용
        if reduction == 'mean':
            mc_loss = mc_loss.mean()
            epistemic_loss = epistemic_loss.mean()
            variance_loss = variance_loss.mean()
            entropy_loss = entropy_loss.mean()
        elif reduction == 'sum':
            mc_loss = mc_loss.sum()
            epistemic_loss = epistemic_loss.sum()
            variance_loss = variance_loss.sum()
            entropy_loss = entropy_loss.sum()
            
        return {
            'mc_loss': mc_loss,
            'epistemic_loss': epistemic_loss,
            'variance_loss': variance_loss,
            'entropy_loss': entropy_loss,
            'adaptive_weights': {'alpha': alpha, 'beta': beta, 'gamma': gamma}
        }
    
    def _compute_epistemic_loss(self, mc_tensor: torch.Tensor) -> torch.Tensor:
        """
        Epistemic Uncertainty Loss 계산
        
        Epistemic Loss = Var[E[p(y|x,θ)]] = 예측 분포들 간의 분산
        """
        try:
            # MC 샘플들의 평균 예측: (B, N, C)
            mean_pred = mc_tensor.mean(dim=0)
            
            # 각 MC 샘플과 평균 간의 분산 계산
            epistemic_variance = ((mc_tensor - mean_pred.unsqueeze(0)) ** 2).mean(dim=0)
                
                # 분산 값 안전장치
            epistemic_variance = torch.clamp(epistemic_variance, 1e-8, 1e6)
            
            # Box regression과 classification 분리
            box_epistemic = epistemic_variance[..., :4]  # bbox coordinates
            cls_epistemic = epistemic_variance[..., 5:]  # class probabilities
            
            # Temperature scaling 적용
            box_loss = (box_epistemic / self.temperature).sum(dim=-1)
            cls_loss = (cls_epistemic / self.temperature).sum(dim=-1)
        
            total_loss = box_loss + cls_loss  # (B, N)
            
            # NaN/Inf 체크
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                print(f"⚠️  Invalid epistemic loss detected: {total_loss}")
                total_loss = torch.zeros_like(total_loss)
            
            return total_loss
            
        except Exception as e:
            print(f"⚠️  Epistemic loss calculation failed: {e}")
            return torch.zeros(mc_tensor.shape[1], mc_tensor.shape[2], device=mc_tensor.device)
    
    def _compute_predictive_variance_loss(self, mc_tensor: torch.Tensor) -> torch.Tensor:
        """
        Predictive Variance Loss 계산
        
        예측 분산을 직접적으로 최소화하여 consistent predictions 유도
        """
        try:
            # 각 위치별 MC 샘플 분산 계산
            pred_variance = mc_tensor.var(dim=0)  # (B, N, C)
            
            # 분산 값 안전장치
            pred_variance = torch.clamp(pred_variance, 1e-8, 1e6)
        
            # Box와 Class 분리하여 가중치 적용
            box_variance = pred_variance[..., :4].sum(dim=-1)  # bbox variance
            obj_variance = pred_variance[..., 4]               # objectness variance  
            cls_variance = pred_variance[..., 5:].sum(dim=-1)  # class variance
            
            # 가중치: box > class > objectness (bbox 정확도 우선)
            weighted_variance = 2.0 * box_variance + 1.5 * cls_variance + 1.0 * obj_variance
        
            # NaN/Inf 체크
            if torch.isnan(weighted_variance).any() or torch.isinf(weighted_variance).any():
                print(f"⚠️  Invalid variance loss detected: {weighted_variance}")
                weighted_variance = torch.zeros_like(weighted_variance)
        
            return weighted_variance  # (B, N)
    
        except Exception as e:
            print(f"⚠️  Variance loss calculation failed: {e}")
            return torch.zeros(mc_tensor.shape[1], mc_tensor.shape[2], device=mc_tensor.device)
    
    def _compute_adaptive_weights(
        self, 
        epistemic_loss: torch.Tensor,
        variance_loss: torch.Tensor, 
        entropy_loss: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        학습 진행에 따른 적응적 가중치 계산
        
        초기: 높은 불확실성 → variance/entropy 중시
        후기: 낮은 불확실성 → epistemic uncertainty 중시
        """
        # Running statistics 업데이트
        with torch.no_grad():
            momentum = 0.99
            self.running_epistemic_mean = self.running_epistemic_mean * momentum + epistemic_loss.mean().detach() * (1-momentum)
            self.running_variance_mean = self.running_variance_mean * momentum + variance_loss.mean().detach() * (1-momentum)
            self.running_entropy_mean = self.running_entropy_mean * momentum + entropy_loss.mean().detach() * (1-momentum)
            self.num_updates += 1
            
        # 정규화된 loss magnitudes
        eps = 1e-6
        epistemic_norm = self.running_epistemic_mean / (self.running_epistemic_mean + eps)
        variance_norm = self.running_variance_mean / (self.running_variance_mean + eps)  
        entropy_norm = self.running_entropy_mean / (self.running_entropy_mean + eps)
        
        # 적응적 가중치 계산 (inverse weighting + curriculum learning)
        progress = min(self.num_updates / 1000.0, 1.0)  # 1000 업데이트까지 curriculum
        
        # 초기에는 variance/entropy 중시, 후기에는 epistemic 중시
        alpha = self.alpha * (0.5 + 0.5 * progress)      # 0.5 → 1.0
        beta = self.beta * (1.5 - 0.5 * progress)        # 1.5 → 1.0  
        gamma = self.gamma * (1.2 - 0.2 * progress)      # 1.2 → 1.0
        
        return float(alpha), float(beta), float(gamma)
    
    def get_uncertainty_quality_metrics(
        self, 
        mc_predictions: List[torch.Tensor],
        confidence_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        MC Loss의 불확실성 품질 평가 메트릭
        """
        if len(mc_predictions) < 2:
            return {}
            
        mc_tensor = torch.stack(mc_predictions, dim=0)
        
        with torch.no_grad():
            # 1. Prediction Consistency (낮을수록 좋음)
            pred_std = mc_tensor.std(dim=0).mean().item()
            
            # 2. Confidence Calibration
            mean_pred = mc_tensor.mean(dim=0)
            obj_conf = torch.sigmoid(mean_pred[..., 4])
            high_conf_ratio = (obj_conf > confidence_threshold).float().mean().item()
            
            # 3. Epistemic vs Aleatoric 분리도
            epistemic_var = mc_tensor.var(dim=0).mean().item()
            mean_pred_var = mean_pred.var().item()
            separation_ratio = epistemic_var / (mean_pred_var + 1e-8)
            
            return {
                'prediction_consistency': pred_std,
                'high_confidence_ratio': high_conf_ratio,
                'epistemic_aleatoric_separation': separation_ratio,
                'total_epistemic_uncertainty': epistemic_var
            } 