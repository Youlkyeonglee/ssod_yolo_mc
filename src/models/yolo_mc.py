from typing import Optional, Union, Dict, Any, List
import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
import yaml
import torch.nn.functional as F

class YOLOWithMCDropout(nn.Module):
    """MC Dropout이 적용된 YOLO 모델"""
    
    def __init__(
        self,
        model_name: str = "yolov8m",
        dropout_rate: float = 0.1,
        task: str = "detect",
        pretrained: bool = True
    ):
        """
        Args:
            model_name: YOLO 모델 이름 (e.g., "yolov8n", "yolov8s", "yolov8m")
            dropout_rate: Dropout 비율
            task: 수행할 작업 (detect, segment, pose)
            pretrained: 사전 학습된 가중치 사용 여부
        """
        super().__init__()
        self.model = YOLO(model_name)
        if not pretrained:
            self.model.reset_weights()
        
        # Dropout 레이어 추가
        self.dropout_rate = dropout_rate
        self._add_dropout_layers()
        
        # 클래스 수 설정
        self.num_classes = 80  # COCO 데이터셋 기준
        
        # 학습 모드로 설정
        self.train()
    
    def _add_dropout_layers(self):
        """모델의 각 컨볼루션 레이어 다음에 Dropout 추가"""
        for module in self.model.model.modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(lambda m, _, output: F.dropout(
                    output,
                    p=self.dropout_rate,
                    training=self.training
                ))
    
    def _process_predictions(self, predictions: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """예측 결과의 클래스 수를 일관되게 처리
        
        Args:
            predictions: 모델의 예측 결과
            
        Returns:
            처리된 예측 결과
        """
        if isinstance(predictions, (list, tuple)):
            processed_preds = []
            for pred in predictions:
                # 현재 클래스 수 확인
                curr_num_classes = pred.shape[-1] - 5
                
                if curr_num_classes != self.num_classes:
                    # print(f"Adjusting number of classes from {curr_num_classes} to {self.num_classes}")
                    # 새로운 텐서 생성
                    new_shape = list(pred.shape)
                    new_shape[-1] = self.num_classes + 5
                    new_pred = torch.zeros(new_shape, device=pred.device, dtype=pred.dtype)
                    
                    # 바운딩 박스와 신뢰도 점수 복사
                    new_pred[..., :5] = pred[..., :5]
                    
                    # 클래스 점수 복사 (가능한 만큼)
                    min_classes = min(curr_num_classes, self.num_classes)
                    new_pred[..., 5:5+min_classes] = pred[..., 5:5+min_classes]
                    
                    processed_preds.append(new_pred)
                else:
                    processed_preds.append(pred)
            return processed_preds
        else:
            curr_num_classes = predictions.shape[-1] - 5
            if curr_num_classes != self.num_classes:
                # print(f"Adjusting number of classes from {curr_num_classes} to {self.num_classes}")
                # 새로운 텐서 생성
                new_shape = list(predictions.shape)
                new_shape[-1] = self.num_classes + 5
                new_pred = torch.zeros(new_shape, device=predictions.device, dtype=predictions.dtype)
                
                # 바운딩 박스와 신뢰도 점수 복사
                new_pred[..., :5] = predictions[..., :5]
                
                # 클래스 점수 복사 (가능한 만큼)
                min_classes = min(curr_num_classes, self.num_classes)
                new_pred[..., 5:5+min_classes] = predictions[..., 5:5+min_classes]
                
                return new_pred
            return predictions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파
        
        Args:
            x: 입력 이미지 텐서 (B, C, H, W)
        
        Returns:
            예측 결과
        """
        # 입력 텐서를 device로 이동하고 float32로 변환
        x = x.to(self.model.device, dtype=torch.float32)
        
        # 학습 중이거나 MC Dropout이 활성화된 경우
        if self.training:
            # 모델을 학습 모드로 설정
            self.model.model.train()
            # forward 패스 수행
            predictions = self.model.model(x)
            
            # 예측 결과 처리
            predictions = self._process_predictions(predictions)
            
            # 디버깅 정보 출력
            if isinstance(predictions, (list, tuple)):
                for i, pred in enumerate(predictions):
                    num_classes = pred.shape[-1] - 5
                    # print(f"Feature map {i} shape: {pred.shape}, num_classes: {num_classes}")
            else:
                num_classes = predictions.shape[-1] - 5
                # print(f"Prediction shape: {predictions.shape}, num_classes: {num_classes}")
            
            return predictions
        else:
            # 평가 모드에서는 일반적인 추론 수행
            self.model.model.eval()
            with torch.no_grad():
                predictions = self.model.model(x)
                
                # 예측 결과 처리
                predictions = self._process_predictions(predictions)
                
                # 디버깅 정보 출력
                if isinstance(predictions, (list, tuple)):
                    for i, pred in enumerate(predictions):
                        num_classes = pred.shape[-1] - 5
                        print(f"Feature map {i} shape: {pred.shape}, num_classes: {num_classes}")
                else:
                    num_classes = predictions.shape[-1] - 5
                    print(f"Prediction shape: {predictions.shape}, num_classes: {num_classes}")
                
                return predictions
    
    def train(self, mode: bool = True):
        """학습/추론 모드 설정"""
        self.training = mode
        self.model.model.train(mode)
        return self
    
    def eval(self):
        """추론 모드 설정"""
        self.train(False)
        return self
    
    def predict(
        self,
        source: Union[str, Path, torch.Tensor],
        conf: float = 0.25,
        device: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        객체 탐지 수행
        
        Args:
            source: 입력 이미지 또는 경로
            conf: 신뢰도 임계값
            device: 실행 디바이스
            **kwargs: 추가 매개변수
        
        Returns:
            탐지 결과
        """
        return self.model.predict(
            source=source,
            conf=conf,
            device=device,
            **kwargs
        )

    def predict_multiple(self, x: torch.Tensor, num_samples: int = 1) -> List[torch.Tensor]:
        """MC Dropout을 사용한 다중 추론
        
        Args:
            x: 입력 이미지 텐서
            num_samples: 샘플링 횟수
        
        Returns:
            예측 결과 리스트
        """
        predictions = []
        self.train()  # Dropout 활성화
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        return predictions 