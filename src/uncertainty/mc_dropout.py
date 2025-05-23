from typing import List, Dict, Any, Union, Optional
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class MCDropoutDetector:
    """Monte Carlo Dropout을 사용한 객체 탐지 불확실성 추정"""
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 10,
        dropout_rate: float = 0.1,
        box_std_threshold: float = 0.1,
        entropy_threshold: float = 0.5,
        conf_threshold: float = 0.25
    ):
        """
        Args:
            model: 기본 객체 탐지 모델
            num_samples: MC Dropout 샘플링 횟수
            dropout_rate: Dropout 비율
            box_std_threshold: 박스 좌표 표준편차 임계값
            entropy_threshold: 클래스 엔트로피 임계값
            conf_threshold: 신뢰도 임계값
        """
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        self.box_std_threshold = box_std_threshold
        self.entropy_threshold = entropy_threshold
        self.conf_threshold = conf_threshold
        
        self._enable_dropout()
    
    def _enable_dropout(self):
        """모델의 모든 Dropout 레이어를 추론 시에도 활성화"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Dropout을 활성화 상태로 유지
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        image: Union[str, torch.Tensor],
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        MC Dropout을 사용하여 불확실성을 포함한 예측 수행
        
        Args:
            image: 입력 이미지 (경로 또는 텐서)
            device: 실행 디바이스
        
        Returns:
            예측 결과와 불확실성 측정값을 포함한 딕셔너리
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        # 이미지를 텐서로 변환
        if isinstance(image, str):
            # 이미지 로드 및 전처리 로직 추가 필요
            pass
        elif isinstance(image, torch.Tensor):
            image = image.to(device)
        
        # 여러 번의 추론 수행
        predictions = []
        self.model.train()  # MC Dropout 활성화
        
        try:
            # 첫 번째 예측으로 출력 형식 확인
            first_pred = self.model(image)
            if isinstance(first_pred, (list, tuple)):
                num_classes = first_pred[0].shape[-1] - 5  # 5 = x,y,w,h,conf
                # print(f"Detected number of classes: {num_classes}")
            else:
                num_classes = first_pred.shape[-1] - 5
                # print(f"Detected number of classes: {num_classes}")
            
            predictions.append(first_pred)
            
            # 나머지 예측 수행
            for _ in range(self.num_samples - 1):
                pred = self.model(image)
                predictions.append(pred)
            
            # 예측 결과 처리
            processed_predictions = []
            for pred in predictions:
                if isinstance(pred, (list, tuple)):
                    # 여러 feature map 결합
                    batch_pred = []
                    for p in pred:
                        # 텐서 차원 출력
                        # print(f"Feature map shape: {p.shape}")
                        # 클래스 수 확인
                        num_classes = p.shape[-1] - 5
                        # print(f"Number of classes in feature map: {num_classes}")
                        # (batch, anchors, grid_h, grid_w, 5+num_classes) -> (batch, -1, 5+num_classes)
                        reshaped = p.view(p.shape[0], -1, p.shape[-1])
                        batch_pred.append(reshaped)
                    pred = torch.cat(batch_pred, dim=1)
                processed_predictions.append(pred)
            
            # 모든 예측을 결합
            try:
                all_predictions = torch.stack(processed_predictions)  # (num_samples, batch, N, 5+num_classes)
                # print(f"Combined predictions shape: {all_predictions.shape}")
                
                # 신뢰도 기반 필터링
                conf_scores = all_predictions[..., 4]
                confident_mask = conf_scores > self.conf_threshold
                
                # 박스와 클래스 분리
                boxes = all_predictions[..., :4]  # (num_samples, batch, N, 4)
                class_scores = all_predictions[..., 5:]  # (num_samples, batch, N, num_classes)
                
                # 평균과 불확실성 계산
                mean_boxes = torch.mean(boxes, dim=0)  # (batch, N, 4)
                box_std = torch.std(boxes, dim=0)  # (batch, N, 4)
                
                # 클래스 엔트로피 계산
                mean_class_probs = torch.softmax(torch.mean(class_scores, dim=0), dim=-1)
                eps = 1e-10
                class_entropy = -torch.sum(mean_class_probs * torch.log(mean_class_probs + eps), dim=-1)
                
                # 결과 반환
                results = []
                for batch_idx in range(mean_boxes.shape[0]):
                    # 배치별 마스크 생성
                    batch_mask = (box_std[batch_idx].mean(dim=-1) < self.box_std_threshold) & \
                                (class_entropy[batch_idx] < self.entropy_threshold) & \
                                confident_mask[0, batch_idx]
                    
                    # 클래스 예측 확률이 가장 높은 클래스 선택
                    class_probs = mean_class_probs[batch_idx]
                    predicted_classes = torch.argmax(class_probs, dim=-1)
                    
                    # 필터링된 결과 저장
                    filtered_boxes = mean_boxes[batch_idx][batch_mask]
                    filtered_scores = conf_scores[0, batch_idx][batch_mask]
                    filtered_classes = predicted_classes[batch_mask]
                    
                    results.append({
                        'boxes': filtered_boxes,
                        'scores': filtered_scores,
                        'labels': filtered_classes,
                        'box_std': box_std[batch_idx][batch_mask],
                        'class_entropy': class_entropy[batch_idx][batch_mask]
                    })
                
                return results
                
            except RuntimeError as e:
                print(f"Error stacking predictions: {str(e)}")
                return None
            
        except Exception as e:
            print(f"Error in MC Dropout prediction: {str(e)}")
            print(f"Prediction shapes:")
            for i, p in enumerate(predictions):
                if isinstance(p, (list, tuple)):
                    print(f"Sample {i} (list):", [x.shape for x in p])
                    if i == 0:
                        print(f"First prediction feature dimensions:", p[0].shape[-1])
                        print(f"First prediction example:", p[0][0, 0, 0])
                else:
                    print(f"Sample {i}:", p.shape)
            return None
    
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
    
    def _compute_class_entropy(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """클래스 예측의 엔트로피 계산
        
        Args:
            predictions: 여러 번의 클래스 예측 결과 리스트
        
        Returns:
            클래스 예측의 엔트로피
        """
        if not predictions:
            return torch.zeros(0)
        
        # 클래스 예측 확률 추출 및 softmax 적용
        class_probs = torch.stack([torch.softmax(pred, dim=-1) for pred in predictions])
        
        # 평균 확률 계산
        mean_probs = torch.mean(class_probs, dim=0)
        
        # 엔트로피 계산 (확률이 0인 경우 처리)
        eps = 1e-10
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=-1)
        
        return entropy
    
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