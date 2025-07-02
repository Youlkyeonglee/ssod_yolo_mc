from typing import List, Dict, Any, Union, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from datetime import datetime
import torch.nn.functional as F
import logging

class MCDropoutDetector:
    """Monte Carlo Dropout을 사용한 객체 탐지 불확실성 추정"""
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 10,
        dropout_rate: float = 0.1,
        box_std_threshold: float = 0.1,
        entropy_threshold: float = 0.5,
        conf_threshold: float = 0.25,
        save_dir: Optional[Path] = None
    ):
        """
        Args:
            model: 기본 객체 탐지 모델
            num_samples: MC Dropout 샘플링 횟수
            dropout_rate: Dropout 비율
            box_std_threshold: 박스 좌표 표준편차 임계값
            entropy_threshold: 클래스 엔트로피 임계값
            conf_threshold: 신뢰도 임계값
            save_dir: 결과 저장 디렉토리
        """
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        self.box_std_threshold = box_std_threshold
        self.entropy_threshold = entropy_threshold
        self.conf_threshold = conf_threshold
        
        # 결과 저장 디렉토리 설정
        if save_dir is None:
            save_dir = Path("runs/train/default/mcdropout")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    

        # 데이터 수집을 위한 변수들
        self.prediction_history = []
        self.uncertainty_history = []
        self.baseline_predictions = []
        
        self._enable_dropout()
    
    def _enable_dropout(self):
        """모델의 모든 Dropout 레이어를 추론 시에도 활성화"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Dropout을 활성화 상태로 유지

    def save_prediction_analysis(self, epoch: int = None):
        """예측 분포 분석 결과를 저장 - 주석처리: 컴퓨터 멈춤 방지"""
        print("MC Dropout analysis saving is disabled to prevent system freeze")
        return
        
        # if not self.prediction_history or not self.uncertainty_history:
        #     print("No prediction data to analyze")
        #     return
        
        # # 에포크별 저장 디렉토리 생성
        # if epoch is not None:
        #     epoch_dir = self.save_dir / f"epoch_{epoch}"
        # else:
        #     epoch_dir = self.save_dir / "final"
        # epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # # 1. 예측 분산 분포 히스토그램
        # self._plot_variance_distribution(epoch_dir)
        
        # # 2. 클래스 엔트로피 분포
        # self._plot_entropy_distribution(epoch_dir)
        
        # # 3. 신뢰도-정확도 캘리브레이션 곡선
        # self._plot_calibration_curve(epoch_dir)
        
        # # 4. Baseline vs MC Dropout 비교
        # self._plot_baseline_comparison(epoch_dir)
        
        # # 5. 불확실성 통계 저장
        # self._save_uncertainty_statistics(epoch_dir)
        
        # # 6. Raw 데이터 저장
        # self._save_raw_data(epoch_dir)
        
        # print(f"MC Dropout analysis saved to {epoch_dir}")

    def _plot_variance_distribution(self, save_dir: Path):
        """예측 분산 분포 히스토그램 생성 및 저장"""
        plt.figure(figsize=(12, 8))
        
        # Box variance 히스토그램 (각 좌표별로)
        all_box_stds = []
        for uncertainty_batch in self.uncertainty_history:
            for result in uncertainty_batch:
                if 'box_std' in result and len(result['box_std']) > 0:
                    box_stds = result['box_std'].cpu().numpy()
                    all_box_stds.extend(box_stds.flatten())
        
        if all_box_stds:
            plt.subplot(2, 2, 1)
            plt.hist(all_box_stds, bins=50, alpha=0.7, color='blue', edgecolor='black')
            plt.title('Box Coordinate Variance Distribution')
            plt.xlabel('Standard Deviation')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # 통계 정보 추가
            mean_std = np.mean(all_box_stds)
            plt.axvline(mean_std, color='red', linestyle='--', label=f'Mean: {mean_std:.4f}')
            plt.axvline(self.box_std_threshold, color='orange', linestyle='--', 
                       label=f'Threshold: {self.box_std_threshold}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'variance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_entropy_distribution(self, save_dir: Path):
        """클래스 엔트로피 분포 히스토그램 생성 및 저장"""
        plt.figure(figsize=(12, 6))
        
        # Class entropy 히스토그램
        all_entropies = []
        all_box_stds = []
        for uncertainty_batch in self.uncertainty_history:
            for result in uncertainty_batch:
                if 'class_entropy' in result and len(result['class_entropy']) > 0:
                    entropies = result['class_entropy'].cpu().numpy()
                    all_entropies.extend(entropies.flatten())
                
                if 'box_std' in result and len(result['box_std']) > 0:
                    box_stds = result['box_std'].cpu().numpy()
                    all_box_stds.extend(box_stds.flatten())
        
        if all_entropies:
            plt.subplot(1, 2, 1)
            plt.hist(all_entropies, bins=50, alpha=0.7, color='green', edgecolor='black')
            plt.title('Class Entropy Distribution')
            plt.xlabel('Entropy')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # 통계 정보 추가
            mean_entropy = np.mean(all_entropies)
            plt.axvline(mean_entropy, color='red', linestyle='--', label=f'Mean: {mean_entropy:.4f}')
            plt.axvline(self.entropy_threshold, color='orange', linestyle='--', 
                       label=f'Threshold: {self.entropy_threshold}')
            plt.legend()
            
            # 엔트로피 vs 분산 scatter plot
            if all_box_stds:
                plt.subplot(1, 2, 2)
                # 동일한 길이로 맞추기
                min_len = min(len(all_entropies), len(all_box_stds))
                plt.scatter(all_entropies[:min_len], all_box_stds[:min_len], 
                           alpha=0.5, s=10)
                plt.xlabel('Class Entropy')
                plt.ylabel('Box Variance')
                plt.title('Entropy vs Variance Correlation')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'entropy_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_calibration_curve(self, save_dir: Path):
        """신뢰도-정확도 캘리브레이션 곡선 생성"""
        plt.figure(figsize=(10, 8))
        
        # MC Dropout 신뢰도와 실제 정확도 계산
        confidences = []
        uncertainties = []
        
        for uncertainty_batch in self.uncertainty_history:
            for result in uncertainty_batch:
                if 'scores' in result and len(result['scores']) > 0:
                    scores = result['scores'].cpu().numpy()
                    confidences.extend(scores)
                
                if 'box_std' in result and len(result['box_std']) > 0:
                    box_stds = result['box_std'].cpu().numpy().mean(axis=1)
                    uncertainties.extend(box_stds)
        
        if confidences and uncertainties:
            # 신뢰도 구간별 불확실성 평균 계산
            confidence_bins = np.linspace(0, 1, 11)
            bin_uncertainties = []
            bin_centers = []
            
            for i in range(len(confidence_bins) - 1):
                mask = (np.array(confidences) >= confidence_bins[i]) & \
                       (np.array(confidences) < confidence_bins[i + 1])
                if mask.sum() > 0:
                    bin_uncertainties.append(np.mean(np.array(uncertainties)[mask]))
                    bin_centers.append((confidence_bins[i] + confidence_bins[i + 1]) / 2)
            
            plt.plot(bin_centers, bin_uncertainties, 'o-', label='MC Dropout')
            plt.xlabel('Confidence Score')
            plt.ylabel('Average Uncertainty')
            plt.title('Confidence vs Uncertainty Calibration')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'calibration_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_baseline_comparison(self, save_dir: Path):
        """Baseline vs MC Dropout 예측 분포 비교"""
        if not self.baseline_predictions:
            return
        
        plt.figure(figsize=(15, 10))
        
        # 1. 예측 신뢰도 분포 비교
        plt.subplot(2, 3, 1)
        mc_confidences = []
        baseline_confidences = []
        
        for uncertainty_batch in self.uncertainty_history:
            for result in uncertainty_batch:
                if 'scores' in result and len(result['scores']) > 0:
                    scores = result['scores'].cpu().numpy()
                    mc_confidences.extend(scores)
        
        for baseline_batch in self.baseline_predictions:
            if isinstance(baseline_batch, list):
                for result in baseline_batch:
                    if 'scores' in result and len(result['scores']) > 0:
                        scores = result['scores'].cpu().numpy() if torch.is_tensor(result['scores']) else result['scores']
                        baseline_confidences.extend(scores)
        
        if mc_confidences and baseline_confidences:
            plt.hist(baseline_confidences, bins=30, alpha=0.5, label='Baseline', color='red')
            plt.hist(mc_confidences, bins=30, alpha=0.5, label='MC Dropout', color='blue')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Confidence Distribution Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. 검출 수 비교
        plt.subplot(2, 3, 2)
        mc_detection_counts = [len(result.get('scores', [])) for batch in self.uncertainty_history for result in batch]
        baseline_detection_counts = [len(result.get('scores', [])) for batch in self.baseline_predictions for result in batch if isinstance(batch, list)]
        
        if mc_detection_counts and baseline_detection_counts:
            plt.hist(baseline_detection_counts, bins=20, alpha=0.5, label='Baseline', color='red')
            plt.hist(mc_detection_counts, bins=20, alpha=0.5, label='MC Dropout', color='blue')
            plt.xlabel('Number of Detections')
            plt.ylabel('Frequency')
            plt.title('Detection Count Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 3. 예측 분산 감소율 계산
        plt.subplot(2, 3, 3)
        if hasattr(self, 'variance_reduction_rate'):
            plt.bar(['Baseline', 'MC Dropout'], [1.0, 1.0 - self.variance_reduction_rate])
            plt.ylabel('Relative Variance')
            plt.title('Prediction Variance Reduction')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _save_uncertainty_statistics(self, save_dir: Path):
        """불확실성 통계 정보를 JSON으로 저장"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'mc_dropout_config': {
                'num_samples': self.num_samples,
                'dropout_rate': self.dropout_rate,
                'box_std_threshold': self.box_std_threshold,
                'entropy_threshold': self.entropy_threshold,
                'conf_threshold': self.conf_threshold
            },
            'statistics': {}
        }
        
        # Box variance 통계
        all_box_stds = []
        for uncertainty_batch in self.uncertainty_history:
            for result in uncertainty_batch:
                if 'box_std' in result and len(result['box_std']) > 0:
                    box_stds = result['box_std'].cpu().numpy()
                    all_box_stds.extend(box_stds.flatten())
        
        if all_box_stds:
            stats['statistics']['box_variance'] = {
                'mean': float(np.mean(all_box_stds)),
                'std': float(np.std(all_box_stds)),
                'min': float(np.min(all_box_stds)),
                'max': float(np.max(all_box_stds)),
                'median': float(np.median(all_box_stds)),
                'percentiles': {
                    '25': float(np.percentile(all_box_stds, 25)),
                    '75': float(np.percentile(all_box_stds, 75)),
                    '95': float(np.percentile(all_box_stds, 95))
                },
                'filtered_ratio': float(np.mean(np.array(all_box_stds) < self.box_std_threshold))
            }
        
        # Class entropy 통계
        all_entropies = []
        for uncertainty_batch in self.uncertainty_history:
            for result in uncertainty_batch:
                if 'class_entropy' in result and len(result['class_entropy']) > 0:
                    entropies = result['class_entropy'].cpu().numpy()
                    all_entropies.extend(entropies.flatten())
        
        if all_entropies:
            stats['statistics']['class_entropy'] = {
                'mean': float(np.mean(all_entropies)),
                'std': float(np.std(all_entropies)),
                'min': float(np.min(all_entropies)),
                'max': float(np.max(all_entropies)),
                'median': float(np.median(all_entropies)),
                'filtered_ratio': float(np.mean(np.array(all_entropies) < self.entropy_threshold))
            }
        
        # Detection count 통계
        detection_counts = [len(result.get('scores', [])) for batch in self.uncertainty_history for result in batch]
        if detection_counts:
            stats['statistics']['detection_counts'] = {
                'mean': float(np.mean(detection_counts)),
                'std': float(np.std(detection_counts)),
                'total_detections': int(np.sum(detection_counts)),
                'total_images': len(detection_counts)
            }
        
        # 예측 분산 감소율 계산 (baseline과 비교)
        if hasattr(self, 'variance_reduction_rate'):
            stats['statistics']['variance_reduction_rate'] = float(self.variance_reduction_rate)
        
        # JSON 저장
        with open(save_dir / 'uncertainty_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)

    def _save_raw_data(self, save_dir: Path):
        """Raw 예측 데이터를 pickle로 저장"""
        raw_data = {
            'prediction_history': self.prediction_history,
            'uncertainty_history': self.uncertainty_history,
            'baseline_predictions': self.baseline_predictions,
            'config': {
                'num_samples': self.num_samples,
                'dropout_rate': self.dropout_rate,
                'box_std_threshold': self.box_std_threshold,
                'entropy_threshold': self.entropy_threshold,
                'conf_threshold': self.conf_threshold
            }
        }
        
        with open(save_dir / 'raw_prediction_data.pkl', 'wb') as f:
            pickle.dump(raw_data, f)

    def add_baseline_prediction(self, baseline_result):
        """Baseline 예측 결과 추가 (비교용)"""
        self.baseline_predictions.append(baseline_result)

    def calculate_variance_reduction_rate(self):
        """Baseline 대비 예측 분산 감소율 계산"""
        if not self.baseline_predictions or not self.uncertainty_history:
            return 0.0
        
        # MC Dropout 예측 분산
        mc_variances = []
        for uncertainty_batch in self.uncertainty_history:
            for result in uncertainty_batch:
                if 'box_std' in result and len(result['box_std']) > 0:
                    variances = result['box_std'].cpu().numpy() ** 2
                    mc_variances.extend(variances.flatten())
        
        if not mc_variances:
            return 0.0
        
        mc_mean_variance = np.mean(mc_variances)
        
        # Baseline은 단일 예측이므로 분산이 0이라고 가정하거나
        # 여러 이미지에서의 예측 분산을 계산
        baseline_variance = 1.0  # 정규화된 baseline 분산
        
        # 분산 감소율 계산
        reduction_rate = (baseline_variance - mc_mean_variance) / baseline_variance
        self.variance_reduction_rate = max(0.0, reduction_rate)
        
        return self.variance_reduction_rate

    def create_summary_report(self, save_dir: Path = None):
        """전체 실험에 대한 요약 보고서 생성 - 주석처리: 컴퓨터 멈춤 방지"""
        print("MC Dropout summary report generation is disabled to prevent system freeze")
        return
        
        # if save_dir is None:
        #     save_dir = self.save_dir
        
        # # 1. 예측 분포 분석 저장
        # self.save_prediction_analysis()
        
        # # 2. 분산 감소율 계산
        # variance_reduction = self.calculate_variance_reduction_rate()
        
        # # 3. 요약 보고서 생성
        # summary = {
        #     'experiment_summary': {
        #         'total_predictions': len(self.uncertainty_history),
        #         'total_baseline_predictions': len(self.baseline_predictions),
        #         'variance_reduction_rate': variance_reduction,
        #         'mc_dropout_effectiveness': variance_reduction > 0.1  # 10% 이상 개선
        #     },
        #     'recommendations': []
        # }
        
        # # 권장사항 생성
        # if variance_reduction > 0.15:
        #     summary['recommendations'].append("MC Dropout이 효과적으로 작동하고 있습니다. 현재 설정을 유지하세요.")
        # elif variance_reduction > 0.05:
        #     summary['recommendations'].append("MC Dropout 효과가 미미합니다. 샘플링 횟수나 dropout rate를 조정해보세요.")
        # else:
        #     summary['recommendations'].append("MC Dropout 효과가 거의 없습니다. 모델 구조나 파라미터를 재검토하세요.")
        
        # with open(save_dir / 'experiment_summary.json', 'w') as f:
        #     json.dump(summary, f, indent=2)
        
        # print(f"MC Dropout 실험 요약 보고서가 {save_dir}에 저장되었습니다.")
        # print(f"예측 분산 감소율: {variance_reduction:.2%}")

    @torch.no_grad()
    def predict_with_uncertainty_legacy(
        self,
        image: Union[str, torch.Tensor],
        device: Optional[str] = None,
        save_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        MC Dropout을 사용하여 불확실성을 포함한 예측 수행
        
        Args:
            image: 입력 이미지 (경로 또는 텐서)
            device: 실행 디바이스
            save_predictions: 예측 결과를 히스토리에 저장할지 여부
        
        Returns:
            예측 결과와 불확실성 측정값을 포함한 딕셔너리
        """
        # 모델이 있는 디바이스 자동 감지
        try:
            # DDP 모델인 경우 module 속성 사용
            if hasattr(self.model, 'module'):
                model_device = next(self.model.module.parameters()).device
            else:
                model_device = next(self.model.parameters()).device
            
            if device is None:
                device = model_device
            elif isinstance(device, str) and device != str(model_device):
                print(f"⚠️  Device mismatch in legacy: model on {model_device}, requested {device}")
                device = model_device  # 모델 디바이스로 강제 설정
        except Exception as e:
            print(f"❌ Device setup error in legacy: {e}")
            # 기본 디바이스 사용
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 이미지를 텐서로 변환
        if isinstance(image, str):
            # 이미지 로드 및 전처리 로직 추가 필요
            pass
        elif isinstance(image, torch.Tensor):
            try:
                image = image.to(device)
            except Exception as e:
                print(f"❌ Image device transfer error: {e}")
                device = image.device  # 이미지가 있는 디바이스 사용
        
        # 여러 번의 추론 수행
        predictions = []
        self.model.train()  # MC Dropout 활성화
        
        try:
            for _ in range(self.num_samples):
                # 안전한 모델 호출 (DDP/DataParallel 지원)
                try:
                    # 모델과 이미지가 같은 디바이스에 있는지 확인
                    if hasattr(self.model, 'module'):
                        # DDP 또는 DataParallel 모델
                        model_device = next(self.model.module.parameters()).device
                        actual_model = self.model.module
                    else:
                        model_device = next(self.model.parameters()).device
                        actual_model = self.model
                    
                    # 이미지를 모델 디바이스로 이동
                    if image.device != model_device:
                        print(f"🔄 Moving image from {image.device} to model device {model_device}")
                        image = image.to(model_device)
                    
                    # 모델 호출
                    pred_output = self.model(image)
                    
                except RuntimeError as device_error:
                    if "device" in str(device_error).lower():
                        print(f"🚨 Device error in MC Dropout: {device_error}")
                        # 이미지 디바이스로 모델을 이동시도 (위험하지만 최후 수단)
                        try:
                            print(f"🔄 Moving model to image device: {image.device}")
                            self.model = self.model.to(image.device)
                            pred_output = self.model(image)
                        except Exception as move_error:
                            print(f"❌ Failed to move model: {move_error}")
                            raise device_error
                    else:
                        raise device_error
                
                # 딕셔너리 형태인 경우 predictions 키에서 실제 예측값 추출
                if isinstance(pred_output, dict):
                    pred = pred_output['predictions']
                else:
                    pred = pred_output
                
                predictions.append(pred)
            
            # 예측 결과 처리
            processed_predictions = []
            for pred in predictions:
                if isinstance(pred, (list, tuple)):
                    # 여러 feature map 결합
                    batch_pred = []
                    expected_classes = None
                    m_count = 0
                    for i, p in enumerate(pred):
                        # 텐서 차원 확인
                        current_classes = p.shape[-1] - 5
                        
                        # 첫 번째 feature map에서 예상 클래스 수 설정
                        if expected_classes is None:
                            expected_classes = current_classes
                        
                        # 클래스 수가 일치하지 않는 경우 디버깅 정보 출력 후 스킵
                        if current_classes != expected_classes:
                            print(f"Warning: Feature map {i} has {current_classes} classes, expected {expected_classes}. Skipping this feature map.")
                            print(f"  Feature map shape: {p.shape}")
                            print(f"  Current classes: {current_classes}, Expected: {expected_classes}")
                            continue
                        else:
                            m_count += 1
                            # print(f"How many time same shape: {m_count}.")
                            # print(f"Feature map shape: {p.shape}")
                            # print(f"Current classes: {current_classes}, Expected: {expected_classes}")
                        # (batch, anchors, grid_h, grid_w, 5+num_classes) -> (batch, -1, 5+num_classes)
                        reshaped = p.view(p.shape[0], -1, p.shape[-1])
                        batch_pred.append(reshaped)
                    
                    # 유효한 feature map이 있는 경우에만 concatenate
                    if batch_pred:
                        pred = torch.cat(batch_pred, dim=1)
                    else:
                        print("Warning: No valid feature maps found. Using empty tensor.")
                        # 빈 텐서 생성 (배치 크기와 예상 클래스 수 사용)
                        if expected_classes is not None:
                            pred = torch.zeros(p.shape[0], 0, 5 + expected_classes).to(p.device)
                        else:
                            pred = torch.zeros(1, 0, 96).to(p.device)  # COCO 기본값
                
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
                    
                    # 안전한 argmax 처리: 클래스 수 체크
                    num_classes = class_probs.shape[-1]
                    predicted_classes = torch.argmax(class_probs, dim=-1)
                    
                    # 클래스 인덱스 안전성 체크
                    max_class_id = predicted_classes.max().item() if predicted_classes.numel() > 0 else -1
                    if max_class_id >= num_classes:
                        print(f"Warning: predicted class {max_class_id} >= num_classes {num_classes}")
                        # 유효 범위로 클램핑
                        predicted_classes = torch.clamp(predicted_classes, 0, num_classes - 1)
                    
                    # 추가 안전성 체크: 음수 클래스 처리
                    predicted_classes = torch.clamp(predicted_classes, 0, max(num_classes - 1, 0))
                    
                    # 필터링된 결과 저장
                    filtered_boxes = mean_boxes[batch_idx][batch_mask]
                    filtered_scores = conf_scores[0, batch_idx][batch_mask]
                    filtered_classes = predicted_classes[batch_mask]
                    
                    # 필터링된 클래스도 다시 한번 체크
                    if filtered_classes.numel() > 0:
                        max_filtered_class = filtered_classes.max().item()
                        if max_filtered_class >= num_classes:
                            print(f"Warning: filtered class {max_filtered_class} >= num_classes {num_classes}")
                            filtered_classes = torch.clamp(filtered_classes, 0, num_classes - 1)
                    
                    result = {
                        'boxes': filtered_boxes,
                        'scores': filtered_scores,
                        'labels': filtered_classes,
                        'box_std': box_std[batch_idx][batch_mask],
                        'class_entropy': class_entropy[batch_idx][batch_mask]
                    }
                    results.append(result)
                
                # 예측 히스토리에 저장 - 메모리 절약을 위해 제한적으로 저장
                if save_predictions and len(self.uncertainty_history) < 10:  # 최대 10개만 저장
                    self.uncertainty_history.append(results)
                    # Raw predictions는 저장하지 않음 (메모리 절약)
                    # self.prediction_history.append([pred.detach().cpu() for pred in processed_predictions])
                
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
                    print(f"Sample {i} last dimensions:", [x.shape[-1] for x in p])
                    print(f"Sample {i} classes per feature map:", [x.shape[-1] - 5 for x in p])
                    if i == 0:
                        print(f"First prediction feature dimensions:", p[0].shape[-1])
                        if len(p[0].shape) >= 3:
                            print(f"First prediction example shape:", p[0][0, 0, :5])
                else:
                    print(f"Sample {i}:", p.shape)
                    print(f"Sample {i} classes:", p.shape[-1] - 5 if len(p.shape) > 0 else "Unknown")
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

    def calculate_consistency_loss(
        self,
        predictions: List[torch.Tensor],
        loss_type: str = "mean_centered"
    ) -> torch.Tensor:
        """
        MC Dropout 예측들 간의 일관성 손실 계산
        
        Args:
            predictions: MC Dropout 샘플들의 예측 결과
            loss_type: "pairwise", "mean_centered", "variance_based"
        
        Returns:
            consistency_loss: 예측 간 일관성 손실
        """
        if len(predictions) < 2:
            return torch.tensor(0.0)
        
        if loss_type == "pairwise":
            # 모든 샘플 쌍 간의 MSE
            consistency_loss = 0.0
            num_samples = len(predictions)
            
            for i in range(num_samples):
                for j in range(i+1, num_samples):
                    consistency_loss += F.mse_loss(predictions[i], predictions[j])
            
            # 정규화
            consistency_loss = consistency_loss / (num_samples * (num_samples - 1) / 2)
            
        elif loss_type == "mean_centered":
            # 평균 기준 일관성 손실 (더 효율적)
            stacked_preds = torch.stack(predictions)  # (num_samples, ...)
            mean_pred = torch.mean(stacked_preds, dim=0)
            
            consistency_loss = 0.0
            for pred in predictions:
                consistency_loss += F.mse_loss(pred, mean_pred)
            
            consistency_loss = consistency_loss / len(predictions)
            
        elif loss_type == "variance_based":
            # 분산 기반 일관성 손실
            stacked_preds = torch.stack(predictions)  # (num_samples, ...)
            variance = torch.var(stacked_preds, dim=0)
            consistency_loss = torch.mean(variance)
            
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        return consistency_loss
    
    def predict_with_consistency_loss(
        self,
        image: Union[str, torch.Tensor],
        device: Optional[str] = None,
        loss_type: str = "mean_centered"
    ) -> Dict[str, Any]:
        """
        불확실성 예측과 함께 consistency loss 계산
        
        Returns:
            결과 딕셔너리에 'consistency_loss' 키 추가
        """
        # 기존 예측 수행
        result = self.predict_with_uncertainty_legacy(image, device, save_predictions=False)
        
        if result is None:
            return None
        
        # MC Dropout 예측들 수집
        predictions = []
        self.model.train()
        
        # 이미지 전처리
        if isinstance(image, str):
            from PIL import Image
            import torchvision.transforms as transforms
            
            pil_image = Image.open(image).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = transform(pil_image).unsqueeze(0)
        
        if device:
            image = image.to(device)
        
        # MC Dropout 샘플링
        for _ in range(self.num_samples):
            with torch.no_grad():
                pred_output = self.model(image)
                
                if isinstance(pred_output, dict):
                    pred = pred_output['predictions']
                else:
                    pred = pred_output
                
                # 예측을 일관된 형태로 변환
                if isinstance(pred, (list, tuple)):
                    # 여러 feature map을 하나로 결합
                    batch_pred = []
                    for p in pred:
                        reshaped = p.view(p.shape[0], -1, p.shape[-1])
                        batch_pred.append(reshaped)
                    pred = torch.cat(batch_pred, dim=1)
                
                predictions.append(pred)
        
        # Consistency Loss 계산
        try:
            consistency_loss = self.calculate_consistency_loss(predictions, loss_type)
            result['consistency_loss'] = consistency_loss.item()
        except Exception as e:
            print(f"Error calculating consistency loss: {e}")
            result['consistency_loss'] = 0.0
        
        return result 

    def calculate_cross_view_consistency_loss(
        self,
        teacher_predictions: List[Any],
        student_predictions: List[torch.Tensor],
        weight: float = 0.1,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        obj_weight: float = 1.0
    ) -> torch.Tensor:
        """Teacher vs Student Cross-View Consistency Loss 계산 (Classification + Regression Loss 적용)
        
        Args:
            teacher_predictions: Teacher 모델의 MC Dropout 예측 결과
            student_predictions: Student 모델의 MC Dropout 예측 결과 
            weight: Cross-view consistency loss 전체 가중치
            cls_weight: Classification loss 가중치
            reg_weight: Regression (bbox) loss 가중치  
            obj_weight: Objectness loss 가중치
            
        Returns:
            Cross-view consistency loss (cls + reg + obj)
        """
        if not teacher_predictions or not student_predictions:
            return torch.tensor(0.0)
        
        try:
            # Teacher 예측들의 평균 계산
            if isinstance(teacher_predictions[0], dict):
                # Teacher 예측이 dict 형태인 경우 (예: {'predictions': tensor})
                teacher_tensors = []
                for pred in teacher_predictions:
                    if isinstance(pred, dict) and 'predictions' in pred:
                        teacher_tensors.append(pred['predictions'])
                    elif torch.is_tensor(pred):
                        teacher_tensors.append(pred)
                
                if teacher_tensors:
                    teacher_mean = torch.mean(torch.stack(teacher_tensors), dim=0)
                else:
                    return torch.tensor(0.0)
            else:
                # Teacher 예측이 tensor 형태인 경우
                teacher_mean = torch.mean(torch.stack(teacher_predictions), dim=0)
            
            # Student 예측들의 평균 계산
            student_mean = torch.mean(torch.stack(student_predictions), dim=0)
            
            # 크기가 다른 경우 처리
            if teacher_mean.shape != student_mean.shape:
                # Teacher와 Student의 feature map 크기가 다른 경우
                # 더 작은 크기로 맞춤
                min_size = min(teacher_mean.shape[-1], student_mean.shape[-1])
                teacher_mean = teacher_mean[..., :min_size]
                student_mean = student_mean[..., :min_size]
            
            # YOLO 예측 형식: [x, y, w, h, objectness, class1, class2, ...]
            # 예측 차원 확인
            if teacher_mean.shape[-1] < 5:
                print(f"Warning: Prediction dimension too small: {teacher_mean.shape[-1]}")
                return torch.tensor(0.0)
            
            # 1. Regression Loss (Bounding Box Coordinates)
            # Box coordinates: [x, y, w, h] (indices 0-3)
            teacher_boxes = teacher_mean[..., :4]  # (batch, N, 4)
            student_boxes = student_mean[..., :4]  # (batch, N, 4)
            
            # Smooth L1 Loss for bounding box regression
            reg_loss = F.smooth_l1_loss(student_boxes, teacher_boxes.detach(), reduction='mean')
            
            # 2. Objectness Loss (Object Confidence)
            # Objectness score: index 4
            teacher_obj = teacher_mean[..., 4]  # (batch, N)
            student_obj = student_mean[..., 4]  # (batch, N)
            
            # Binary Cross Entropy for objectness
            teacher_obj_sigmoid = torch.sigmoid(teacher_obj.detach())
            obj_loss = F.binary_cross_entropy_with_logits(
                student_obj, teacher_obj_sigmoid, reduction='mean'
            )
            
            # 3. Classification Loss (Class Probabilities)
            # Class scores: indices 5 onwards
            if teacher_mean.shape[-1] > 5:
                teacher_cls = teacher_mean[..., 5:]  # (batch, N, num_classes)
                student_cls = student_mean[..., 5:]  # (batch, N, num_classes)
                
                # Cross Entropy Loss for classification
                # Teacher의 softmax 확률을 target으로 사용 (Knowledge Distillation 방식)
                teacher_cls_probs = F.softmax(teacher_cls.detach(), dim=-1)
                student_cls_log_probs = F.log_softmax(student_cls, dim=-1)
                
                # KL Divergence Loss (더 안정적인 확률 분포 일치)
                cls_loss = F.kl_div(
                    student_cls_log_probs, 
                    teacher_cls_probs, 
                    reduction='batchmean'
                )
            else:
                cls_loss = torch.tensor(0.0, device=teacher_mean.device)
            
            # 4. 총 Cross-View Consistency Loss 계산
            total_consistency_loss = (
                reg_weight * reg_loss + 
                obj_weight * obj_loss + 
                cls_weight * cls_loss
            ) * weight
            
            # 디버깅 정보 (선택적)
            if torch.isnan(total_consistency_loss) or torch.isinf(total_consistency_loss):
                print(f"Warning: Invalid cross-view consistency loss detected")
                print(f"  Reg loss: {reg_loss.item():.6f}")
                print(f"  Obj loss: {obj_loss.item():.6f}")
                print(f"  Cls loss: {cls_loss.item():.6f}")
                return torch.tensor(0.0, device=teacher_mean.device)
            
            # Loss 정보를 딕셔너리로 반환 (디버깅용)
            loss_info = {
                'total': total_consistency_loss,
                'reg_loss': reg_loss,
                'obj_loss': obj_loss, 
                'cls_loss': cls_loss,
                'weights': {
                    'total_weight': weight,
                    'reg_weight': reg_weight,
                    'obj_weight': obj_weight,
                    'cls_weight': cls_weight
                }
            }
            
            # 메인 loss만 반환하되, 필요시 loss_info 접근 가능하도록 속성 추가
            total_consistency_loss.loss_info = loss_info
            
            return total_consistency_loss
            
        except Exception as e:
            print(f"Error in calculate_cross_view_consistency_loss: {e}")
            return torch.tensor(0.0) 
    
    def generate_reliable_pseudo_labels(
        self, 
        all_predictions: torch.Tensor, 
        reliability_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """다층 신뢰도 평가 시스템으로 고품질 pseudo label 생성
        
        Args:
            all_predictions: MC Dropout 예측 결과 (num_samples, batch, N, 5+num_classes)
            reliability_threshold: 신뢰도 임계값 (0.0-1.0)
        
        Returns:
            고품질 pseudo label 리스트
        """
        
        def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
            """두 박스 간 IoU 계산 (xywh 형식)"""
            # xywh -> xyxy 변환
            def xywh_to_xyxy(box):
                x_center, y_center, width, height = box
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                return torch.stack([x1, y1, x2, y2])
            
            box1_xyxy = xywh_to_xyxy(box1)
            box2_xyxy = xywh_to_xyxy(box2)
            
            # Intersection 계산
            x1 = torch.max(box1_xyxy[0], box2_xyxy[0])
            y1 = torch.max(box1_xyxy[1], box2_xyxy[1])
            x2 = torch.min(box1_xyxy[2], box2_xyxy[2])
            y2 = torch.min(box1_xyxy[3], box2_xyxy[3])
            
            intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
            
            # Union 계산
            area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
            area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
            union = area1 + area2 - intersection
            
            iou = intersection / (union + 1e-10)
            return iou.item()
        
        def calculate_reliability_score(mc_samples_single_det: torch.Tensor) -> Dict[str, float]:
            """단일 detection에 대한 종합 신뢰도 점수 계산"""
            
            # Level 1: Geometric Consistency (박스 기하학적 일관성)
            boxes = mc_samples_single_det[:, :4]
            box_std = torch.std(boxes, dim=0)
            geometric_score = torch.exp(-torch.mean(box_std))  # 분산이 낮을수록 높은 점수
            
            # Level 2: Semantic Consistency (클래스 의미적 일관성)  
            class_logits = mc_samples_single_det[:, 5:]
            class_probs = torch.softmax(class_logits, dim=-1)
            
            # 클래스 예측의 엔트로피 (낮을수록 확실함)
            mean_probs = torch.mean(class_probs, dim=0)
            entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10))
            semantic_score = torch.exp(-entropy)
            
            # Level 3: Confidence Stability (신뢰도 안정성)
            conf_scores = torch.sigmoid(mc_samples_single_det[:, 4])
            conf_cv = torch.std(conf_scores) / (torch.mean(conf_scores) + 1e-10)  # Coefficient of Variation
            stability_score = torch.exp(-conf_cv)
            
            # Level 4: Inter-sample Agreement (샘플 간 합의도)
            pairwise_agreements = []
            num_samples = mc_samples_single_det.shape[0]
            
            for i in range(num_samples):
                for j in range(i+1, num_samples):
                    # 박스 IoU
                    box_iou = calculate_iou(boxes[i], boxes[j])
                    
                    # 클래스 일치도
                    class_i = torch.argmax(class_logits[i])
                    class_j = torch.argmax(class_logits[j])
                    class_agree = float(class_i == class_j)
                    
                    # 신뢰도 유사도
                    conf_sim = 1.0 - abs(conf_scores[i] - conf_scores[j])
                    
                    agreement = (box_iou + class_agree + conf_sim) / 3.0
                    pairwise_agreements.append(agreement)
            
            agreement_score = torch.mean(torch.stack(pairwise_agreements)) if pairwise_agreements else 0.0
            
            # 가중 평균으로 최종 신뢰도 점수 계산
            weights = [0.3, 0.3, 0.2, 0.2]  # geometric, semantic, stability, agreement
            final_score = (weights[0] * geometric_score + 
                          weights[1] * semantic_score + 
                          weights[2] * stability_score + 
                          weights[3] * agreement_score)
            
            return {
                'final_score': final_score.item(),
                'geometric_score': geometric_score.item(),
                'semantic_score': semantic_score.item(), 
                'stability_score': stability_score.item(),
                'agreement_score': agreement_score.item() if isinstance(agreement_score, torch.Tensor) else agreement_score
            }
        
        # 모든 detection에 대해 신뢰도 평가
        num_samples, batch, N, _ = all_predictions.shape
        reliable_labels = []
        
        for batch_idx in range(batch):
            for det_idx in range(N):
                mc_samples = all_predictions[:, batch_idx, det_idx, :]
                
                # 기본 신뢰도 체크
                mean_conf = torch.mean(torch.sigmoid(mc_samples[:, 4]))
                if mean_conf < 0.3:
                    continue
                
                # 종합 신뢰도 점수 계산
                reliability_scores = calculate_reliability_score(mc_samples)
                
                if reliability_scores['final_score'] > reliability_threshold:
                    # 최종 예측값 계산
                    mean_box = torch.mean(mc_samples[:, :4], dim=0)
                    mean_class_logits = torch.mean(mc_samples[:, 5:], dim=0)
                    predicted_class = torch.argmax(mean_class_logits)
                    
                    reliable_labels.append({
                        'batch_idx': batch_idx,
                        'det_idx': det_idx,
                        'box': mean_box,
                        'class': predicted_class.item(),
                        'confidence': mean_conf.item(),
                        'reliability_scores': reliability_scores,
                        'quality_tier': 'high' if reliability_scores['final_score'] > 0.9 else 'medium'
                    })
        
        return reliable_labels
    
    def predict_with_uncertainty(
        self,
        image: Union[str, torch.Tensor],
        device: Optional[str] = None,
        save_predictions: bool = True,
        reliability_threshold: Optional[float] = None,
        config: Optional[dict] = None
    ) -> Dict[str, Any]:
        """다층 신뢰도 평가 시스템을 적용한 불확실성 예측 (V5 통합)
        
        Args:
            image: 입력 이미지
            device: 디바이스
            save_predictions: 예측 저장 여부
            reliability_threshold: 신뢰도 임계값 (None이면 config에서 읽기)
            config: 설정 딕셔너리 (reliability_threshold 읽기용)
        
        Returns:
            다층 신뢰도 평가가 적용된 예측 결과
        """
        # reliability_threshold 설정
        if reliability_threshold is None:
            if config is not None:
                reliability_threshold = config.get('training', {}).get(
                    'semi_supervised', {}
                ).get('uncertainty', {}).get('reliability_threshold', 0.8)
            else:
                reliability_threshold = 0.8  # 기본값
        # 기존 predict_with_uncertainty_legacy 호출
        base_result = self.predict_with_uncertainty_legacy(image, device, save_predictions=False)
        
        if base_result is None:
            return None
        
        # MC Dropout 다시 수행하여 all_predictions 얻기
        if isinstance(image, str):
            pass  # 이미지 로드 로직 필요
        elif isinstance(image, torch.Tensor):
            # 모델이 있는 디바이스 자동 감지
            try:
                # DDP 모델인 경우 module 속성 사용
                if hasattr(self.model, 'module'):
                    model_device = next(self.model.module.parameters()).device
                else:
                    model_device = next(self.model.parameters()).device
                
                if device is None:
                    device = model_device
                elif isinstance(device, str) and device != str(model_device):
                    print(f"⚠️  Device mismatch: model on {model_device}, requested {device}")
                    device = model_device  # 모델 디바이스로 강제 설정
                
                image = image.to(device)
            except Exception as e:
                print(f"❌ Device setup error: {e}")
                # 이미지가 이미 있는 디바이스 사용
                device = image.device
                print(f"🔄 Using image device: {device}")
        
        predictions = []
        self.model.train()  # MC Dropout 활성화
        
        try:
            for _ in range(self.num_samples):
                # 안전한 모델 호출 (DDP/DataParallel 지원)
                try:
                    # 모델과 이미지가 같은 디바이스에 있는지 확인
                    if hasattr(self.model, 'module'):
                        # DDP 또는 DataParallel 모델
                        model_device = next(self.model.module.parameters()).device
                    else:
                        model_device = next(self.model.parameters()).device
                    
                    # 이미지를 모델 디바이스로 이동
                    if image.device != model_device:
                        print(f"🔄 Moving image from {image.device} to model device {model_device}")
                        image = image.to(model_device)
                    
                    # 모델 호출
                    pred_output = self.model(image)
                    
                except RuntimeError as device_error:
                    if "device" in str(device_error).lower():
                        print(f"🚨 Device error in MC Dropout v2: {device_error}")
                        # 이미지 디바이스로 모델을 이동시도 (위험하지만 최후 수단)
                        try:
                            print(f"🔄 Moving model to image device: {image.device}")
                            self.model = self.model.to(image.device)
                            pred_output = self.model(image)
                        except Exception as move_error:
                            print(f"❌ Failed to move model: {move_error}")
                            raise device_error
                    else:
                        raise device_error
                
                if isinstance(pred_output, dict):
                    pred = pred_output['predictions']
                else:
                    pred = pred_output
                
                # 예측 처리 (기존 로직과 동일)
                processed_predictions = []
                if isinstance(pred, (list, tuple)):
                    batch_pred = []
                    expected_classes = None
                    
                    for p in pred:
                        current_classes = p.shape[-1] - 5
                        
                        if expected_classes is None:
                            expected_classes = current_classes
                        
                        if current_classes != expected_classes:
                            continue
                        
                        reshaped = p.view(p.shape[0], -1, p.shape[-1])
                        batch_pred.append(reshaped)
                    
                    if batch_pred:
                        pred = torch.cat(batch_pred, dim=1)
                    else:
                        if expected_classes is not None:
                            pred = torch.zeros(p.shape[0], 0, 5 + expected_classes).to(p.device)
                        else:
                            pred = torch.zeros(1, 0, 96).to(p.device)
                
                predictions.append(pred)
            
            # all_predictions 생성
            all_predictions = torch.stack(predictions)  # (num_samples, batch, N, 5+num_classes)
            
            # V5 신뢰도 평가 시스템 적용
            reliable_labels = self.generate_reliable_pseudo_labels(
                all_predictions, reliability_threshold
            )
            
            # 기존 결과에 V5 결과 추가
            enhanced_results = []
            for batch_idx, batch_result in enumerate(base_result):
                # 해당 배치의 신뢰도 라벨 필터링
                batch_reliable_labels = [
                    label for label in reliable_labels 
                    if label['batch_idx'] == batch_idx
                ]
                
                # 신뢰도 기반 필터링된 결과 생성
                if batch_reliable_labels:
                    filtered_boxes = torch.stack([label['box'] for label in batch_reliable_labels])
                    filtered_scores = torch.tensor([label['confidence'] for label in batch_reliable_labels])
                    filtered_classes = torch.tensor([label['class'] for label in batch_reliable_labels])
                    
                    enhanced_result = {
                        'boxes': filtered_boxes,
                        'scores': filtered_scores,
                        'labels': filtered_classes,
                        'reliable_labels': batch_reliable_labels,
                        'reliability_stats': {
                            'total_detections': len(batch_reliable_labels),
                            'high_quality': len([l for l in batch_reliable_labels if l['quality_tier'] == 'high']),
                            'medium_quality': len([l for l in batch_reliable_labels if l['quality_tier'] == 'medium']),
                            'avg_final_score': sum([l['reliability_scores']['final_score'] for l in batch_reliable_labels]) / len(batch_reliable_labels) if batch_reliable_labels else 0.0
                        }
                    }
                else:
                    # 신뢰도 기준을 통과한 라벨이 없는 경우
                    enhanced_result = {
                        'boxes': torch.zeros((0, 4)),
                        'scores': torch.zeros(0),
                        'labels': torch.zeros(0, dtype=torch.long),
                        'reliable_labels': [],
                        'reliability_stats': {
                            'total_detections': 0,
                            'high_quality': 0,
                            'medium_quality': 0,
                            'avg_final_score': 0.0
                        }
                    }
                
                enhanced_results.append(enhanced_result)
            
            # 예측 히스토리 저장 (제한적)
            if save_predictions and len(self.uncertainty_history) < 10:
                self.uncertainty_history.append(enhanced_results)
            
            return enhanced_results
            
        except Exception as e:
            print(f"Error in predict_with_uncertainty_v5: {str(e)}")
            return base_result 

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
        # MC 샘플들의 평균 예측: (B, N, C)
        mean_pred = mc_tensor.mean(dim=0)
        
        # 각 MC 샘플과 평균 간의 분산 계산
        epistemic_variance = ((mc_tensor - mean_pred.unsqueeze(0)) ** 2).mean(dim=0)
        
        # Box regression과 classification 분리
        box_epistemic = epistemic_variance[..., :4]  # bbox coordinates
        cls_epistemic = epistemic_variance[..., 5:]  # class probabilities
        
        # Temperature scaling 적용
        box_loss = (box_epistemic / self.temperature).sum(dim=-1)
        cls_loss = (cls_epistemic / self.temperature).sum(dim=-1)
        
        return box_loss + cls_loss  # (B, N)
    
    def _compute_predictive_variance_loss(self, mc_tensor: torch.Tensor) -> torch.Tensor:
        """
        Predictive Variance Loss 계산
        
        예측 분산을 직접적으로 최소화하여 consistent predictions 유도
        """
        # 각 위치별 MC 샘플 분산 계산
        pred_variance = mc_tensor.var(dim=0)  # (B, N, C)
        
        # Box와 Class 분리하여 가중치 적용
        box_variance = pred_variance[..., :4].sum(dim=-1)  # bbox variance
        obj_variance = pred_variance[..., 4]               # objectness variance  
        cls_variance = pred_variance[..., 5:].sum(dim=-1)  # class variance
        
        # 가중치: box > class > objectness (bbox 정확도 우선)
        weighted_variance = 2.0 * box_variance + 1.5 * cls_variance + 1.0 * obj_variance
        
        return weighted_variance  # (B, N)
    
    def _compute_entropy_regularization(self, mc_tensor: torch.Tensor) -> torch.Tensor:
        """
        Entropy Regularization 계산
        
        각 MC 샘플의 예측 엔트로피를 정규화하여 confident predictions 유도
        """
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
        
        return mean_entropy + 0.5 * obj_entropy  # (B, N)
    
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
            self.running_epistemic_mean.mul_(momentum).add_(
                epistemic_loss.mean().detach(), alpha=1-momentum
            )
            self.running_variance_mean.mul_(momentum).add_(
                variance_loss.mean().detach(), alpha=1-momentum
            )
            self.running_entropy_mean.mul_(momentum).add_(
                entropy_loss.mean().detach(), alpha=1-momentum
            )
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

 