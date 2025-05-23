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
            return self.model.model(x)
        else:
            # 평가 모드에서는 일반적인 추론 수행
            self.model.model.eval()
            with torch.no_grad():
                return self.model.model(x)
    
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