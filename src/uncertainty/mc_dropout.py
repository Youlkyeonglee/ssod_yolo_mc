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
        self.baseline_predictions = []
        
        self._enable_dropout()
    
    def _enable_dropout(self):
        """모델의 모든 Dropout 레이어를 추론 시에도 활성화"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Dropout을 활성화 상태로 유지

    def update_adaptive_thresholds(self, box_stds: torch.Tensor, class_entropies: torch.Tensor):
        """
        계산된 box_std와 class_entropy 값들을 기반으로 동적 임계값 업데이트
        
        Args:
            box_stds: 현재 배치의 box_std 값들 (N, 4) 또는 (batch, N, 4)
            class_entropies: 현재 배치의 class_entropy 값들 (N,) 또는 (batch, N)
        """
        if not self.adaptive_thresholds:
            return
        
        with torch.no_grad():
            # 텐서 차원 정규화
            if len(box_stds.shape) == 3:  # (batch, N, 4)
                box_stds = box_stds.view(-1, 4)  # (batch*N, 4)
            if len(class_entropies.shape) == 2:  # (batch, N)
                class_entropies = class_entropies.view(-1)  # (batch*N,)
            
            # 유효한 값들만 필터링 (NaN, inf 제거)
            valid_box_mask = torch.isfinite(box_stds).all(dim=-1)
            valid_entropy_mask = torch.isfinite(class_entropies)
            
            if valid_box_mask.any():
                # box_std 평균 계산 (각 좌표별 평균의 평균)
                valid_box_stds = box_stds[valid_box_mask]
                current_box_std_mean = valid_box_stds.mean().item()
                
                # 동적 임계값 업데이트 (모멘텀 적용)
                self.running_box_std_mean = (
                    self.threshold_momentum * self.running_box_std_mean + 
                    (1 - self.threshold_momentum) * current_box_std_mean
                )
                
                # 임계값을 평균값의 일정 비율로 설정 (예: 평균의 80%)
                self.box_std_threshold = self.running_box_std_mean * 0.8
            
            if valid_entropy_mask.any():
                # class_entropy 평균 계산
                valid_entropies = class_entropies[valid_entropy_mask]
                current_entropy_mean = valid_entropies.mean().item()
                
                # 동적 임계값 업데이트 (모멘텀 적용)
                self.running_entropy_mean = (
                    self.threshold_momentum * self.running_entropy_mean + 
                    (1 - self.threshold_momentum) * current_entropy_mean
                )
                
                # 임계값을 평균값의 일정 비율로 설정 (예: 평균의 80%)
                self.entropy_threshold = self.running_entropy_mean * 0.8
            
            self.threshold_update_count += 1
            
            # 디버깅 정보 출력 (처음 몇 번만)
            if self.threshold_update_count <= 5:
                print(f"🔄 동적 임계값 업데이트 #{self.threshold_update_count.item()}:")
                if valid_box_mask.any():
                    print(f"  - box_std: 평균={current_box_std_mean:.4f}, 임계값={self.box_std_threshold:.4f}")
                if valid_entropy_mask.any():
                    print(f"  - entropy: 평균={current_entropy_mean:.4f}, 임계값={self.entropy_threshold:.4f}")

    def get_current_thresholds(self) -> Dict[str, float]:
        """현재 동적 임계값들을 반환"""
        return {
            'box_std_threshold': self.box_std_threshold,
            'entropy_threshold': self.entropy_threshold,
            'update_count': self.threshold_update_count.item(),
            'adaptive_enabled': self.adaptive_thresholds
        }

    def reset_adaptive_thresholds(self, box_std_threshold: float = None, entropy_threshold: float = None):
        """동적 임계값을 초기값으로 리셋"""
        if box_std_threshold is not None:
            self.box_std_threshold = box_std_threshold
            self.running_box_std_mean = torch.tensor(box_std_threshold)
        
        if entropy_threshold is not None:
            self.entropy_threshold = entropy_threshold
            self.running_entropy_mean = torch.tensor(entropy_threshold)
        
        self.threshold_update_count = torch.tensor(0)
        print(f"🔄 동적 임계값 리셋: box_std={self.box_std_threshold:.4f}, entropy={self.entropy_threshold:.4f}")

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
                # print(f"conf_scores: {conf_scores}")
                confident_mask = conf_scores > self.conf_threshold
                # print(f"confident_mask: {confident_mask}")
                # 박스와 클래스 분리
                boxes = all_predictions[..., :4]  # (num_samples, batch, N, 4)
                class_scores = all_predictions[..., 5:]  # (num_samples, batch, N, num_classes)
                
                # 평균과 불확실성 계산
                mean_boxes = torch.mean(boxes, dim=0)  # (batch, N, 4)
                box_std = torch.std(boxes, dim=0)  # (batch, N, 4)
                
                # 박스 좌표 검증 및 수정
                try:
                    # 이미지 크기 추정 (박스 좌표에서)
                    if mean_boxes.numel() > 0:
                        max_coords = mean_boxes.max(dim=0)[0].max(dim=0)[0]
                        estimated_img_size = (int(max_coords[1].item()), int(max_coords[0].item()))
                    else:
                        estimated_img_size = (640, 640)  # 기본값
                    
                    # 박스 좌표 검증
                    for batch_idx in range(mean_boxes.shape[0]):
                        batch_boxes = mean_boxes[batch_idx].cpu().numpy()
                        validated_boxes, valid_mask = validate_box_coordinates(batch_boxes, estimated_img_size)
                        
                        # 유효한 박스만 유지
                        if np.any(valid_mask):
                            mean_boxes[batch_idx] = torch.from_numpy(validated_boxes[valid_mask]).to(mean_boxes.device)
                            box_std[batch_idx] = box_std[batch_idx][valid_mask]
                            confident_mask[0, batch_idx] = confident_mask[0, batch_idx][valid_mask]
                        else:
                            # 모든 박스가 유효하지 않은 경우 빈 텐서로 설정
                            mean_boxes[batch_idx] = torch.empty(0, 4, device=mean_boxes.device)
                            box_std[batch_idx] = torch.empty(0, 4, device=box_std.device)
                            confident_mask[0, batch_idx] = torch.empty(0, dtype=torch.bool, device=confident_mask.device)
                            
                except Exception as e:
                    print(f"⚠️ 박스 좌표 검증 중 오류 발생: {e}")
                    # 검증 실패 시 원본 사용
                
                # 클래스 엔트로피 계산
                mean_class_probs = torch.softmax(torch.mean(class_scores, dim=0), dim=-1)
                eps = 1e-10
                class_entropy = -torch.sum(mean_class_probs * torch.log(mean_class_probs + eps), dim=-1)
                
                # 결과 반환
                results = []
                batch_box_std_list = []
                batch_entropy_list = []
                for batch_idx in range(mean_boxes.shape[0]):
                    # 배치별 마스크 생성 (inplace operation 방지를 위해 복사)
                    print(f"🔍 임계값 비교:")
                    print(f"📦 필터링 전 전체 박스 수: {box_std[batch_idx].shape[0]}")
                    # 기본 필터링 조건 적용
                    batch_mask = (box_std[batch_idx].mean(dim=-1) < self.box_std_threshold) & \
                                (class_entropy[batch_idx] < self.entropy_threshold) & \
                                confident_mask[0, batch_idx]
                    # max_pseudo_labels에 따른 추가 필터링
                    if hasattr(self, 'max_pseudo_labels') and self.max_pseudo_labels > 0:
                        # confidence 점수로 정렬하여 상위 N개만 선택
                        conf_scores_batch = conf_scores[0, batch_idx][batch_mask]
                        if len(conf_scores_batch) > self.max_pseudo_labels:
                            _, top_indices = torch.topk(conf_scores_batch, self.max_pseudo_labels)
                            new_mask = torch.zeros_like(batch_mask)
                            new_mask[torch.where(batch_mask)[0][top_indices]] = True
                            batch_mask = new_mask
                    print(f"✅ 필터링 후 남은 박스 수: {batch_mask.sum().item()}")
                    
                    
                    # 클래스 예측 확률이 가장 높은 클래스 선택
                    class_probs = mean_class_probs[batch_idx].clone()  # 복사하여 inplace 방지
                    
                    # 안전한 argmax 처리: 클래스 수 체크
                    num_classes = class_probs.shape[-1]
                    predicted_classes = torch.argmax(class_probs, dim=-1)
                    
                    # 클래스 인덱스 안전성 체크
                    max_class_id = predicted_classes.max().item() if predicted_classes.numel() > 0 else -1
                    if max_class_id >= num_classes:
                        print(f"Warning: predicted class {max_class_id} >= num_classes {num_classes}")
                        # 유효 범위로 클램핑 (복사하여 inplace 방지)
                        predicted_classes = torch.clamp(predicted_classes.clone(), 0, num_classes - 1)
                    
                    # 추가 안전성 체크: 음수 클래스 처리
                    predicted_classes = torch.clamp(predicted_classes.clone(), 0, max(num_classes - 1, 0))
                    
                    # 필터링된 결과 저장 (복사하여 inplace 방지)
                    filtered_boxes = mean_boxes[batch_idx][batch_mask].clone()
                    filtered_scores = conf_scores[0, batch_idx][batch_mask].clone()
                    filtered_classes = predicted_classes[batch_mask].clone()
                    filtered_box_std = box_std[batch_idx][batch_mask].clone()
                    filtered_class_entropy = class_entropy[batch_idx][batch_mask].clone()
                    
                    # 박스 좌표 유효성 검사 및 클램핑 (마이너스 값 문제 해결)
                    if filtered_boxes.numel() > 0:
                        filtered_boxes = torch.clamp(filtered_boxes, 0.0, 1.0)
                        valid_box_mask = (filtered_boxes[:, 2] >= 0.01) & (filtered_boxes[:, 3] >= 0.01)
                        if not valid_box_mask.all():
                            filtered_boxes = filtered_boxes[valid_box_mask]
                            filtered_scores = filtered_scores[valid_box_mask]
                            filtered_classes = filtered_classes[valid_box_mask]
                            filtered_box_std = filtered_box_std[valid_box_mask]
                            filtered_class_entropy = filtered_class_entropy[valid_box_mask]
                    
                    # 필터링된 클래스도 다시 한번 체크
                    if filtered_classes.numel() > 0:
                        max_filtered_class = filtered_classes.max().item()
                        if max_filtered_class >= num_classes:
                            print(f"Warning: filtered class {max_filtered_class} >= num_classes {num_classes}")
                            filtered_classes = torch.clamp(filtered_classes.clone(), 0, num_classes - 1)
                    
                    result = {
                        'boxes': filtered_boxes,
                        'scores': filtered_scores,
                        'labels': filtered_classes,
                        'box_std': filtered_box_std,
                        'class_entropy': filtered_class_entropy
                    }
                    results.append(result)
                    # 동적 임계값 후보값 저장
                    if filtered_box_std.numel() > 0:
                        batch_box_std_list.append(filtered_box_std.mean().item())
                    if filtered_class_entropy.numel() > 0:
                        batch_entropy_list.append(filtered_class_entropy.mean().item())
                
                # === 동적 임계값 업데이트 ===
                if batch_box_std_list:
                    new_box_std_threshold = float(np.mean(batch_box_std_list))
                    # 안전장치: 임계값이 너무 작아지지 않도록 제한
                    min_box_std_threshold = 0.01  # 최소 임계값
                    self.box_std_threshold = max(new_box_std_threshold, min_box_std_threshold)
                else:
                    # 필터링된 박스가 없는 경우: 전체 박스의 평균을 사용하여 임계값 완화
                    if box_std.numel() > 0:
                        overall_box_std_mean = box_std.mean().item()
                        # 전체 평균의 1.5배로 임계값 설정 (더 관대하게)
                        self.box_std_threshold = overall_box_std_mean * 1.5
                        print(f"⚠️  필터링된 박스 없음 - 전체 평균 기반 임계값 설정: {self.box_std_threshold:.4f}")
                
                if batch_entropy_list:
                    new_entropy_threshold = float(np.mean(batch_entropy_list))
                    # 안전장치: 임계값이 너무 작아지지 않도록 제한
                    min_entropy_threshold = 0.01  # 최소 임계값
                    self.entropy_threshold = max(new_entropy_threshold, min_entropy_threshold)
                else:
                    # 필터링된 박스가 없는 경우: 전체 엔트로피의 평균을 사용하여 임계값 완화
                    if class_entropy.numel() > 0:
                        overall_entropy_mean = class_entropy.mean().item()
                        # 전체 평균의 1.5배로 임계값 설정 (더 관대하게)
                        self.entropy_threshold = overall_entropy_mean * 1.5
                        print(f"⚠️  필터링된 박스 없음 - 전체 평균 기반 임계값 설정: {self.entropy_threshold:.4f}")
                
                print(f"[Dynamic] box_std_threshold: {self.box_std_threshold:.4f}, entropy_threshold: {self.entropy_threshold:.4f}")
                
                # 추가 안전장치: 여전히 필터링된 박스가 없는 경우 임계값을 더 완화
                total_filtered_boxes = sum(len(result['boxes']) for result in results)
                if total_filtered_boxes == 0:
                    print("🚨 모든 배치에서 필터링된 박스가 0개 - 임계값을 더 완화합니다")
                    # 임계값을 2배로 완화
                    self.box_std_threshold *= 2.0
                    self.entropy_threshold *= 2.0
                    print(f"[Emergency] box_std_threshold: {self.box_std_threshold:.4f}, entropy_threshold: {self.entropy_threshold:.4f}")
                    
                    # 재필터링 시도 (선택적)
                    # results = self._refilter_with_relaxed_thresholds(mean_boxes, box_std, class_entropy, conf_scores, predicted_classes, num_classes)
                
                # 예측 히스토리에 저장 - 메모리 절약을 위해 제한적으로 저장
                if save_predictions and len(self.uncertainty_history) < 10:  # 최대 10개만 저장
                    self.uncertainty_history.append(results)
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
        # print("="*80)
        # print("enhanced_results: ", enhanced_results)
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

 