from typing import Optional, Union, Dict, Any, List
import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
import yaml
import torch.nn.functional as F
from .feature_alignment import FeatureAlignmentModule

class YOLOWithMCDropout(nn.Module):
    """MC Dropout이 적용된 YOLO 모델"""
    
    def __init__(
        self,
        model_name: str = "yolov8m",
        dropout_rate: float = 0.1,
        task: str = "detect",
        pretrained: bool = True,
        feature_alignment_enabled: bool = True,
        num_classes: int = 91,
        ema_decay: float = 0.999  # EMA decay rate for teacher update
    ):
        """
        Args:
            model_name: YOLO 모델 이름
            dropout_rate: MC Dropout 비율
            task: 작업 유형
            pretrained: 사전 학습된 가중치 사용 여부
            feature_alignment_enabled: Feature alignment 사용 여부
            num_classes: 클래스 수
            ema_decay: Teacher 모델 EMA 업데이트 decay rate (0.999 권장)
        """
        super().__init__()
        
        # EMA decay rate 저장
        self.ema_decay = ema_decay
        
        # 학생 모델 (학습 가능) - 먼저 생성
        print("🔧 Creating student model...")
        # pre-trained 가중치 없이 모델 생성
        if pretrained:
            self.student_model = YOLO(model_name)
        else:
            # pre-trained 가중치 없이 모델 생성
            self.student_model = YOLO(model_name)
            # 가중치를 랜덤 초기화
            self._initialize_random_weights(self.student_model)
        
        # 모델의 클래스 수 설정
        self._set_model_num_classes(self.student_model, num_classes)
        
        # MC Dropout 적용 (Student에만 적용)
        print("🎯 Adding MC Dropout to student model...")
        self.dropout_rate = dropout_rate
        self._add_dropout_layers(self.student_model.model)
        
        # 교사 모델을 Student의 완전한 복사본으로 생성
        print("👨‍🏫 Creating teacher model as exact copy of student...")
        import copy
        try:
            # 방법 1: Deep copy 시도
            self.teacher_model = copy.deepcopy(self.student_model)
            print("✅ Teacher created via deep copy")
            
        except Exception as e:
            print(f"⚠️  Deep copy failed: {e}")
            print("🔄 Falling back to manual state dict copy...")
            
            # 방법 2: 수동 복사
            if pretrained:
                self.teacher_model = YOLO(model_name)
            else:
                # pre-trained 가중치 없이 모델 생성
                self.teacher_model = YOLO(model_name)
                # 가중치를 랜덤 초기화
                self._initialize_random_weights(self.teacher_model)
            self._set_model_num_classes(self.teacher_model, num_classes)
            
            # Student의 완전한 state dict를 Teacher에 복사
            student_state = self.student_model.model.state_dict()
            
            try:
                self.teacher_model.model.load_state_dict(student_state, strict=True)
                print("✅ Teacher synchronized via state dict (strict=True)")
            except Exception as state_e:
                print(f"⚠️  Strict loading failed: {state_e}")
                try:
                    self.teacher_model.model.load_state_dict(student_state, strict=False)
                    print("✅ Teacher synchronized via state dict (strict=False)")
                except Exception as final_e:
                    print(f"❌ All synchronization methods failed: {final_e}")
                    print("   Teacher and Student may have different structures!")
        
        # Teacher 모델에서 MC Dropout hook 제거 (Teacher는 deterministic하게 동작해야 함)
        print("🔧 Removing MC Dropout hooks from teacher model...")
        self._remove_dropout_hooks(self.teacher_model.model)
        
        # Teacher 모델을 eval 모드로 설정하되, EMA 업데이트를 위해 파라미터는 유지
        self.teacher_model.model.eval()
        for param in self.teacher_model.model.parameters():
            param.requires_grad = False  # Teacher는 직접 gradient로 학습하지 않음
        
        # Feature Alignment Module 초기화
        self.feature_alignment_enabled = feature_alignment_enabled
        if feature_alignment_enabled:
            # YOLOv8m의 실제 feature map 크기에 맞게 설정
            # P3: [32, 192, 80, 80]
            # P4: [32, 384, 40, 40]
            # P5: [32, 576, 20, 20]
            self.feature_alignment = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(192, 192, kernel_size=1),
                    nn.BatchNorm2d(192),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(384, 384, kernel_size=1),
                    nn.BatchNorm2d(384),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(576, 576, kernel_size=1),
                    nn.BatchNorm2d(576),
                    nn.ReLU(inplace=True)
                )
            ])
        else:
            self.feature_alignment = None
        
        # 클래스 수 설정
        self.num_classes = num_classes
        
        # 학습 모드로 설정
        self.train()
        
        # 마지막으로 teacher-student 모델 동기화 확인
        self._verify_model_sync()
        
        # EMA 업데이트 활성화 여부 (파라미터 불일치 시 비활성화)
        self.ema_enabled = self._is_ema_safe()
    
    def _verify_model_sync(self):
        """Teacher와 Student 모델의 파라미터 동기화 상태 확인"""
        print("🔍 Verifying teacher-student model synchronization...")
        
        teacher_params = list(self.teacher_model.model.parameters())
        student_params = list(self.student_model.model.parameters())
        
        if len(teacher_params) != len(student_params):
            print(f"⚠️  Parameter count mismatch: Teacher={len(teacher_params)}, Student={len(student_params)}")
            return False
        
        mismatch_count = 0
        total_params = len(teacher_params)
        
        for i, (t_param, s_param) in enumerate(zip(teacher_params, student_params)):
            if t_param.shape != s_param.shape:
                if mismatch_count < 5:  # 처음 5개만 상세 출력
                    print(f"   Parameter {i}: Teacher {t_param.shape} ≠ Student {s_param.shape}")
                mismatch_count += 1
        
        if mismatch_count > 0:
            print(f"❌ Found {mismatch_count}/{total_params} parameters with shape mismatches")
            print("   EMA updates will be skipped for mismatched parameters")
            return False
        else:
            print(f"✅ All {total_params} parameters have matching shapes")
            return True

    def _is_ema_safe(self):
        """EMA 업데이트가 안전한지 확인"""
        try:
            teacher_params = list(self.teacher_model.model.parameters())
            student_params = list(self.student_model.model.parameters())
            
            if len(teacher_params) != len(student_params):
                print(f"⚠️  EMA disabled: Parameter count mismatch")
                return False
            
            for i, (t_param, s_param) in enumerate(zip(teacher_params, student_params)):
                if t_param.shape != s_param.shape:
                    print(f"⚠️  EMA disabled: Shape mismatch at parameter {i}")
                    return False
            
            print(f"✅ EMA enabled: All parameters compatible")
            return True
            
        except Exception as e:
            print(f"⚠️  EMA disabled: Error checking compatibility - {e}")
            return False

    def _initialize_random_weights(self, yolo_model):
        """YOLO 모델의 가중치를 랜덤 초기화"""
        print("🔄 Initializing random weights for YOLO model...")
        
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # Xavier/Glorot 초기화
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm 초기화
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                # Transpose Conv 초기화
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 모델의 모든 파라미터를 랜덤 초기화
        if hasattr(yolo_model, 'model'):
            yolo_model.model.apply(init_weights)
        else:
            yolo_model.apply(init_weights)
        
        print("✅ Random weight initialization completed")

    def _set_model_num_classes(self, yolo_model, num_classes):
        """YOLO 모델의 클래스 수를 포괄적으로 설정"""
        try:
            # 1. 모델 객체의 nc 속성 설정
            if hasattr(yolo_model, 'model'):
                # 중첩된 model 구조 처리
                if hasattr(yolo_model.model, 'model') and hasattr(yolo_model.model.model, 'nc'):
                    yolo_model.model.model.nc = num_classes
                if hasattr(yolo_model.model, 'nc'):
                    yolo_model.model.nc = num_classes
                    
            # 2. YOLO 인스턴스 자체의 속성 설정
            if hasattr(yolo_model, 'nc'):
                yolo_model.nc = num_classes
                    
            # 3. 모델의 args 설정 (nc는 args에서 제외 - Ultralytics에서 허용하지 않음)
            # if hasattr(yolo_model, 'args'):
            #     if hasattr(yolo_model.args, 'nc'):
            #         yolo_model.args.nc = num_classes
            #     # args가 dict인 경우도 처리
            #     elif isinstance(yolo_model.args, dict):
            #         yolo_model.args['nc'] = num_classes
                    
            # 4. cfg 설정 (YAML 설정)
            if hasattr(yolo_model, 'cfg'):
                if isinstance(yolo_model.cfg, dict):
                    yolo_model.cfg['nc'] = num_classes
                elif hasattr(yolo_model.cfg, 'nc'):
                    yolo_model.cfg.nc = num_classes
                    
            # 5. data 설정 (validation에서 중요)
            if hasattr(yolo_model, 'data') and isinstance(yolo_model.data, dict):
                yolo_model.data['nc'] = num_classes
                
            # 6. trainer 설정 (있는 경우) - args.nc는 제외
            if hasattr(yolo_model, 'trainer') and yolo_model.trainer:
                if hasattr(yolo_model.trainer, 'data') and isinstance(yolo_model.trainer.data, dict):
                    yolo_model.trainer.data['nc'] = num_classes
                # trainer.args.nc는 설정하지 않음 (Ultralytics에서 허용하지 않음)
                    
            print(f"✓ Successfully set model classes to {num_classes}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not fully set model classes: {e}")
            print(f"   Attempting partial configuration...")
            
            # 최소한의 설정이라도 시도
            try:
                if hasattr(yolo_model, 'model') and hasattr(yolo_model.model, 'nc'):
                    yolo_model.model.nc = num_classes
                    print(f"✓ Partial success: set model.nc = {num_classes}")
            except:
                print("❌ Failed to set even basic model.nc")
    
    def _add_dropout_layers(self, model):
        """모델에 MC Dropout 레이어 추가"""
        def dropout_hook(m, _, output):
            return nn.functional.dropout(
                output,
                p=self.dropout_rate,
                training=self.training
            )
        
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                handle = module.register_forward_hook(dropout_hook)
                # hook을 추적할 수 있도록 저장 (나중에 제거할 때 사용)
                if not hasattr(module, '_mc_dropout_hooks'):
                    module._mc_dropout_hooks = []
                module._mc_dropout_hooks.append(handle)
    
    def _remove_dropout_hooks(self, model):
        """모델에서 MC Dropout hook 제거"""
        removed_count = 0
        for module in model.modules():
            if hasattr(module, '_mc_dropout_hooks'):
                for handle in module._mc_dropout_hooks:
                    handle.remove()
                    removed_count += 1
                delattr(module, '_mc_dropout_hooks')
        
        if removed_count > 0:
            print(f"✅ Removed {removed_count} MC Dropout hooks from model")
    
    def _process_predictions(self, predictions: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """예측 결과의 클래스 수를 일관되게 처리
        
        Args:
            predictions: 모델의 예측 결과
            
        Returns:
            처리된 예측 결과
        """
        # None이나 빈 리스트인 경우 그대로 반환
        if predictions is None:
            return predictions
        if isinstance(predictions, (list, tuple)) and len(predictions) == 0:
            return predictions
        if isinstance(predictions, (list, tuple)):
            processed_preds = []
            for pred in predictions:
                # pred가 또 다시 리스트인 경우 재귀적으로 처리
                if isinstance(pred, (list, tuple)):
                    # 네스티드 리스트인 경우 재귀 호출
                    processed_pred = self._process_predictions(pred)
                    processed_preds.append(processed_pred)
                elif isinstance(pred, torch.Tensor):
                    # 텐서인 경우 클래스 수 확인
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
                else:
                    # 예상하지 못한 타입인 경우 그대로 추가
                    print(f"Warning: Unexpected prediction type: {type(pred)}")
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

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 이미지 배치
        
        Returns:
            dict: {
                'predictions': 최종 예측 결과,
                'student_features': 학생 모델의 feature maps,
                'teacher_features': 교사 모델의 feature maps (feature_alignment_enabled=True인 경우),
                'alignment_loss': feature alignment loss (feature_alignment_enabled=True인 경우)
            }
        """
        # 학생 모델 forward
        student_features = []
        x_student = x
        
        # Backbone을 통한 feature extraction
        for i, m in enumerate(self.student_model.model.model):
            if i <= 9:  # backbone layers
                x_student = m(x_student)
                if i in [4, 6, 9]:  # P3, P4, P5에 해당하는 레이어
                    student_features.append(x_student)
        
        # 원본 YOLO 모델로 예측 수행
        predictions = self.student_model.model(x)
        
        # 디버깅: 예측 결과 타입 확인
        # print(f"Original predictions type: {type(predictions)}")
        # if isinstance(predictions, (list, tuple)):
        #     print(f"  List length: {len(predictions)}")
        #     if len(predictions) > 0:
        #         print(f"  First element type: {type(predictions[0])}")
        
        # 예측 결과의 클래스 수 조정
        predictions = self._process_predictions(predictions)
        
        # YOLOv8의 학습 모드에서 loss도 함께 반환되는지 확인
        if self.training and hasattr(self.student_model.model, 'training_step'):
            # 학습 모드에서는 loss 정보도 함께 저장
            self.last_loss_info = getattr(self.student_model.model, 'loss_items', None)
        
        # Feature alignment가 활성화된 경우
        if self.feature_alignment_enabled:
            # 교사 모델의 feature maps 추출 (고정)
            with torch.no_grad():
                teacher_features = []
                x_teacher = x
                for i, m in enumerate(self.teacher_model.model.model):
                    if i <= 9:  # backbone layers
                        x_teacher = m(x_teacher)
                        if i in [4, 6, 9]:  # P3, P4, P5에 해당하는 레이어
                            teacher_features.append(x_teacher)
            
            # Feature alignment loss 계산
            alignment_loss = 0
            for idx, (student_feat, teacher_feat) in enumerate(zip(student_features, teacher_features)):
                # feature map의 shape 출력 (디버깅용)
                # print(f"Level {idx} - Student feature shape: {student_feat.shape}")
                # print(f"Level {idx} - Teacher feature shape: {teacher_feat.shape}")
                
                # Feature alignment 적용
                transformed_student = self.feature_alignment[idx](student_feat)
                alignment_loss += nn.functional.mse_loss(transformed_student, teacher_feat)
            
            alignment_loss = alignment_loss / len(self.feature_alignment)
        else:
            teacher_features = None
            alignment_loss = 0.0
        
        return {
            'predictions': predictions,
            'student_features': student_features,
            'teacher_features': teacher_features,
            'alignment_loss': alignment_loss
        }

    def train(self, mode: bool = True):
        """학습/추론 모드 설정"""
        self.training = mode
        self.student_model.model.train(mode)
        return self
    
    def eval(self):
        """모델을 평가 모드로 설정"""
        self.student_model.model.eval()
        self.teacher_model.model.eval()
        return super().eval()
    
    def update_teacher_ema(self):
        """
        EMA를 사용하여 teacher 모델의 파라미터를 업데이트
        teacher_param = decay * teacher_param + (1 - decay) * student_param
        
        Note: EMA 호환성이 사전에 확인된 경우에만 업데이트 수행
        """
        # EMA가 비활성화된 경우 건너뛰기
        if not getattr(self, 'ema_enabled', False):
            return
        
        try:
            with torch.no_grad():
                teacher_params = list(self.teacher_model.model.parameters())
                student_params = list(self.student_model.model.parameters())
                
                # 안전성을 위한 추가 체크
                if len(teacher_params) != len(student_params):
                    print(f"⚠️  Disabling EMA: Parameter count changed during training")
                    self.ema_enabled = False
                    return
                
                # 모든 파라미터 업데이트 (사전에 호환성 확인됨)
                for teacher_param, student_param in zip(teacher_params, student_params):
                    if teacher_param.shape == student_param.shape:
                        # EMA 공식: θ_teacher = α * θ_teacher + (1 - α) * θ_student
                        teacher_param.data.mul_(self.ema_decay).add_(
                            student_param.data, alpha=1 - self.ema_decay
                        )
                    else:
                        # 런타임에서 크기가 바뀐 경우 EMA 비활성화
                        print(f"⚠️  Disabling EMA: Parameter shape changed during training")
                        self.ema_enabled = False
                        return
                        
        except Exception as e:
            print(f"⚠️  Disabling EMA due to error: {e}")
            self.ema_enabled = False
    
    def reset_teacher_to_student(self):
        """
        Teacher 모델의 파라미터를 현재 student 모델로 초기화
        학습 초기나 특별한 경우에 사용
        """
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_model.model.parameters(),
                self.student_model.model.parameters()
            ):
                # 파라미터 크기가 일치하는지 확인
                if teacher_param.shape != student_param.shape:
                    print(f"⚠️  Shape mismatch in reset - Teacher: {teacher_param.shape}, Student: {student_param.shape}")
                    print(f"   Skipping parameter reset for this layer...")
                    continue
                
                try:
                    teacher_param.data.copy_(student_param.data)
                except RuntimeError as e:
                    print(f"⚠️  Error resetting parameter: {e}")
                    print(f"   Teacher shape: {teacher_param.shape}, Student shape: {student_param.shape}")
                    continue
    
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
        # validation 전에 클래스 수 재설정
        self._set_model_num_classes(self.student_model, self.num_classes)
        
        # 시각화 관련 매개변수 비활성화
        kwargs.update({
            'save': False,  # 시각화 저장 비활성화
            'save_txt': False,  # 텍스트 결과 저장 비활성화
            'save_conf': False,  # 신뢰도 저장 비활성화
            'save_json': False,  # JSON 저장 비활성화
            'plots': False,  # 플롯 생성 비활성화
            'verbose': False  # 상세 출력 비활성화
        })
        
        return self.student_model.predict(
            source=source,
            conf=conf,
            device=device,
            **kwargs
        )
    
    def val(self, **kwargs):
        """
        모델 validation 수행
        
        Args:
            **kwargs: validation 매개변수
        
        Returns:
            validation 결과
        """
        # validation 전에 클래스 수 재설정
        self._set_model_num_classes(self.student_model, self.num_classes)
        
        return self.student_model.val(**kwargs)

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
                # forward에서 이미 _process_predictions가 적용됨
                predictions.append(pred['predictions'])
        
        return predictions 

    def compute_loss(self, predictions, targets):
        """
        YOLO loss 계산
        
        Args:
            predictions: 모델 예측 결과
            targets: 실제 라벨
            
        Returns:
            dict: loss 딕셔너리 (box_loss, cls_loss, obj_loss 포함)
        """
        try:
            # YOLOv8의 loss 계산 함수 사용
            if hasattr(self.student_model.model, 'loss'):
                loss_dict = self.student_model.model.loss(predictions, targets)
                return loss_dict
            elif hasattr(self.student_model.model, 'compute_loss'):
                loss_dict = self.student_model.model.compute_loss(predictions, targets)
                return loss_dict
            else:
                # 기본 MSE loss 사용 (fallback)
                total_loss = nn.functional.mse_loss(predictions, targets)
                return {
                    'box_loss': total_loss * 0.5,
                    'cls_loss': total_loss * 0.3,
                    'obj_loss': total_loss * 0.2
                }
        except Exception as e:
            print(f"Loss 계산 중 오류 발생: {e}")
            # 기본값 반환
            return {
                'box_loss': torch.tensor(0.0, device=predictions.device),
                'cls_loss': torch.tensor(0.0, device=predictions.device), 
                'obj_loss': torch.tensor(0.0, device=predictions.device)
            } 

    def get_loss_info(self):
        """마지막으로 계산된 loss 정보 반환"""
        return getattr(self, 'last_loss_info', None) 