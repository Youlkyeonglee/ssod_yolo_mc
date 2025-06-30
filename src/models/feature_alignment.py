import torch
import torch.nn as nn

class FeatureAlignmentModule(nn.Module):
    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        
        # 학생 feature를 교사 feature 공간으로 변환하는 선형 변환 네트워크
        self.transform = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1),
            nn.BatchNorm2d(teacher_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features (List[torch.Tensor]): 학생 모델의 feature map 리스트
            teacher_features (List[torch.Tensor]): 교사 모델의 feature map 리스트
        
        Returns:
            torch.Tensor: Feature alignment loss (MSE)
        """
        alignment_loss = 0
        for student_feat, teacher_feat in zip(student_features, teacher_features):
            # 학생 feature를 교사 feature 공간으로 변환
            transformed_student = self.transform(student_feat)
            
            # MSE 손실 계산
            loss = nn.functional.mse_loss(transformed_student, teacher_feat)
            alignment_loss += loss
            
        return alignment_loss / len(student_features)  # 평균 손실 반환 