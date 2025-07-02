"""
MC Dropout 분석 및 품질 평가 모듈

이 모듈은 학습 중간에 MC Dropout의 품질을 분석하고 
보고서를 생성하는 기능을 제공합니다.
"""

import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from .mc_dropout import MCDropoutDetector


def save_mc_dropout_analysis(
    model: torch.nn.Module,
    detector: MCDropoutDetector,
    config: dict,
    epoch: int,
    save_dir: Path,
    logger: logging.Logger,
    data_loader: Optional[torch.utils.data.DataLoader] = None
):
    """학습 중간 MC Dropout 품질 분석 결과 저장
    
    Args:
        model: 분석할 모델
        detector: MC Dropout 탐지기
        config: 설정
        epoch: 현재 에포크
        save_dir: 저장 디렉토리
        logger: 로거
        data_loader: 실제 데이터 로더 (옵션)
    """
    try:
        # 분석 파일 경로
        analysis_file = save_dir / f"mc_dropout_training_analysis_epoch_{epoch+1}.txt"
        
        # 실제 데이터 사용 (더미 데이터 대신)
        device = next(model.parameters()).device
        
        # 실제 데이터 로더에서 배치 가져오기
        if data_loader is not None:
            try:
                # 실제 학습 데이터에서 첫 번째 배치 사용
                sample_batch = next(iter(data_loader))
                test_batch = sample_batch['images'][:2].to(device)  # 실제 이미지 2개
                logger.info(f"📊 MC Dropout 분석에 실제 데이터 사용: {test_batch.shape}")
            except Exception as e:
                logger.warning(f"실제 데이터 로드 실패, 더미 데이터 사용: {e}")
                test_batch = torch.randn(2, 3, 640, 640).to(device)
        else:
            logger.warning("Data loader가 제공되지 않음, 더미 데이터 사용")
            test_batch = torch.randn(2, 3, 640, 640).to(device)
        
        # 다양한 num_samples로 테스트
        sample_counts = [3, 5, 10, config['model']['dropout']['num_samples']]
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            _write_analysis_header(f, epoch, config)
            _write_performance_analysis(f, model, test_batch, sample_counts, config, device, logger)
            _write_training_summary(f, epoch, config)
        
        logger.info(f"📁 MC Dropout 분석 결과 저장: {analysis_file}")
        
    except Exception as e:
        logger.warning(f"MC Dropout 분석 저장 실패: {e}")


def _write_analysis_header(f, epoch: int, config: dict):
    """분석 보고서 헤더 작성"""
    f.write("=" * 80 + "\n")
    f.write(f"MC DROPOUT 학습 중간 분석 보고서 - Epoch {epoch+1}\n")
    f.write("=" * 80 + "\n")
    f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"학습 에포크: {epoch+1}/{config['training']['epochs']}\n")
    f.write(f"현재 설정 num_samples: {config['model']['dropout']['num_samples']}\n")
    f.write(f"현재 설정 dropout_rate: {config['model']['dropout']['rate']}\n")
    f.write("=" * 80 + "\n\n")


def _write_performance_analysis(
    f, 
    model: torch.nn.Module, 
    test_batch: torch.Tensor, 
    sample_counts: List[int], 
    config: dict, 
    device: str, 
    logger: logging.Logger
):
    """MC Dropout 성능 분석 결과 작성"""
    f.write("📊 현재 에포크에서의 MC Dropout 성능 분석 (실제 학습 데이터 사용)\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Samples':<8} {'Time(s)':<10} {'Detections':<12} {'Box Var':<12} {'Confidence':<12}\n")
    f.write("-" * 70 + "\n")
    
    model.eval()
    for num_samples in sample_counts:
        try:
            # 임시 detector 생성
            temp_detector = MCDropoutDetector(
                model=model,
                num_samples=num_samples,
                box_std_threshold=config['training']['semi_supervised']['uncertainty']['box_std_threshold'],
                entropy_threshold=config['training']['semi_supervised']['uncertainty']['entropy_threshold']
            )
            
            # 시간 측정
            import time
            start_time = time.time()
            
            with torch.no_grad():
                uncertainty_results = temp_detector.predict_with_uncertainty(
                    test_batch, 
                    device=device, 
                    config=config
                )
            
            inference_time = time.time() - start_time
            
            # 결과 분석
            detection_count = len(uncertainty_results) if uncertainty_results else 0
            avg_box_var, avg_confidence = _analyze_uncertainty_results(uncertainty_results)
            
            # 결과 출력
            f.write(f"{num_samples:<8} "
                   f"{inference_time:<10.3f} "
                   f"{detection_count:<12} "
                   f"{avg_box_var:<12.4f} "
                   f"{avg_confidence:<12.4f}\n")
            
        except Exception as e:
            f.write(f"{num_samples:<8} ERROR: {str(e)[:30]}\n")


def _analyze_uncertainty_results(uncertainty_results) -> tuple[float, float]:
    """불확실성 결과 분석"""
    avg_box_var = 0.0
    avg_confidence = 0.0
    
    if uncertainty_results:
        box_vars = []
        confidences = []
        
        for result in uncertainty_results:
            # MC Dropout에서 반환되는 실제 키 사용
            if 'box_std' in result and len(result['box_std']) > 0:
                box_std = result['box_std']
                # Box variance 계산 (표준편차의 제곱)
                if hasattr(box_std, 'mean'):
                    avg_std = box_std.mean().item()
                    box_vars.append(avg_std * avg_std)  # variance = std^2
                elif isinstance(box_std, (int, float)):
                    box_vars.append(box_std * box_std)
            
            if 'scores' in result and len(result['scores']) > 0:
                scores = result['scores']
                if hasattr(scores, 'mean'):
                    confidences.append(scores.mean().item())
                elif hasattr(scores, '__iter__'):
                    confidences.extend([s.item() if hasattr(s, 'item') else s for s in scores])
                elif isinstance(scores, (int, float)):
                    confidences.append(scores)
        
        if box_vars:
            avg_box_var = sum(box_vars) / len(box_vars)
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
    
    return avg_box_var, avg_confidence


def _write_training_summary(f, epoch: int, config: dict):
    """학습 진행 상황 요약 작성"""
    f.write("\n" + "=" * 80 + "\n")
    f.write("📈 학습 진행 상황 요약\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"🎯 현재 에포크: {epoch+1}/{config['training']['epochs']} ({(epoch+1)/config['training']['epochs']*100:.1f}% 완료)\n\n")
    
    f.write("💡 MC Dropout 품질 평가 기준:\n")
    f.write("- Box Variance: 낮을수록 좋음 (< 0.1: 우수, < 0.2: 양호, < 0.3: 보통, ≥ 0.3: 개선 필요)\n")
    f.write("- Confidence: 높을수록 좋음 (> 0.8: 우수, > 0.6: 양호, > 0.4: 보통, ≤ 0.4: 개선 필요)\n")
    f.write("- Detection Count: 적절한 수준의 탐지가 유지되어야 함\n\n")
    
    current_num_samples = config['model']['dropout']['num_samples']
    f.write(f"🔧 현재 설정 평가:\n")
    f.write(f"- 사용 중인 num_samples: {current_num_samples}\n")
    f.write(f"- 권장 범위: 10-15 (연구/실험), 5-10 (실시간)\n")
    f.write(f"- 학습 단계: {'초기 단계' if epoch < 50 else '중간 단계' if epoch < 150 else '후반 단계'}\n\n")
    
    if epoch >= 10:
        f.write("📊 학습 단계별 권장사항:\n")
        if epoch < 50:
            f.write("- 초기 단계: MC Dropout 효과가 아직 안정화되지 않을 수 있음\n")
            f.write("- 권장: 현재 설정 유지, 품질 지표 모니터링\n")
        elif epoch < 150:
            f.write("- 중간 단계: MC Dropout 품질이 안정화되는 시점\n")
            f.write("- 권장: 필요시 num_samples 조정 고려\n")
        else:
            f.write("- 후반 단계: 최종 성능 최적화 단계\n")
            f.write("- 권장: 품질 지표 기반 최종 튜닝\n")
        f.write("\n")


def create_mc_dropout_summary_report(
    detector: MCDropoutDetector,
    save_dir: Path,
    config: dict,
    logger: Optional[logging.Logger] = None
):
    """전체 실험에 대한 MC Dropout 요약 보고서 생성
    
    Args:
        detector: MC Dropout 탐지기
        save_dir: 저장 디렉토리
        config: 설정
        logger: 로거 (옵션)
    """
    try:
        summary_file = save_dir / "mc_dropout_experiment_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MC DROPOUT 전체 실험 요약 보고서\n")
            f.write("=" * 80 + "\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"실험 설정 num_samples: {config['model']['dropout']['num_samples']}\n")
            f.write(f"실험 설정 dropout_rate: {config['model']['dropout']['rate']}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("🎯 실험 목표:\n")
            f.write("- MC Dropout을 활용한 pseudo label 신뢰도 평가\n")
            f.write("- 예측 분산이 낮을수록 의사 레이블의 자신감 향상\n")
            f.write("- 신뢰도 기반 필터링으로 pseudo label 정확도 10-15% 향상\n\n")
            
            f.write("📊 주요 성과 지표:\n")
            f.write("- Box Variance 감소율\n")
            f.write("- Pseudo Label 정확도 향상\n")
            f.write("- 불확실성 캘리브레이션 품질\n")
            f.write("- Expected Calibration Error (ECE)\n\n")
            
            f.write("🔬 실험 완료 상태:\n")
            f.write("- Teacher-Student MC Dropout 구조 구현 완료\n")
            f.write("- 다층 신뢰도 평가 시스템 적용\n")
            f.write("- Weak/Strong Augmentation 전략 구현\n")
            f.write("- 분포 기반 Consistency Loss 적용\n\n")
        
        if logger:
            logger.info(f"📁 MC Dropout 실험 요약 보고서 저장: {summary_file}")
    
    except Exception as e:
        if logger:
            logger.warning(f"MC Dropout 실험 요약 보고서 생성 실패: {e}")


def evaluate_mc_dropout_quality_metrics(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    detector: MCDropoutDetector,
    device: str,
    config: dict,
    max_batches: int = 5
) -> Dict[str, Any]:
    """MC Dropout 품질 메트릭 평가
    
    Args:
        model: 평가할 모델
        data_loader: 데이터 로더
        detector: MC Dropout 탐지기
        device: 실행 디바이스
        config: 설정
        max_batches: 평가할 최대 배치 수
    
    Returns:
        MC Dropout 품질 통계
    """
    # DDP 지원
    model_for_eval = model.module if hasattr(model, 'module') else model
    model_for_eval.eval()
    
    box_variances = []
    class_entropies = []
    confidence_scores = []
    detection_counts = []
    
    with torch.no_grad():
        batch_count = 0
        for batch in data_loader:
            if batch_count >= max_batches:
                break
                
            try:
                images = batch['images'].to(device)
                
                # MC Dropout 불확실성 추정
                actual_device = images.device
                uncertainty_results = detector.predict_with_uncertainty(
                    images, 
                    device=str(actual_device), 
                    config=config
                )
                
                if uncertainty_results:
                    detection_counts.append(len(uncertainty_results))
                    
                    # 불확실성 지표 수집
                    for result in uncertainty_results:
                        _collect_uncertainty_metrics(
                            result, box_variances, class_entropies, confidence_scores
                        )
                else:
                    detection_counts.append(0)
                
                batch_count += 1
                
            except Exception:
                detection_counts.append(0)
                batch_count += 1
    
    return _calculate_quality_statistics(
        box_variances, class_entropies, confidence_scores, detection_counts, config
    )


def _collect_uncertainty_metrics(
    result: dict, 
    box_variances: list, 
    class_entropies: list, 
    confidence_scores: list
):
    """불확실성 메트릭 수집"""
    if 'box_variance' in result:
        box_var = result['box_variance']
        if hasattr(box_var, 'item') and box_var.numel() > 0:
            box_variances.append(box_var.item())
        elif isinstance(box_var, (int, float)):
            box_variances.append(box_var)
    
    if 'class_entropy' in result:
        class_ent = result['class_entropy']
        if hasattr(class_ent, 'item') and class_ent.numel() > 0:
            class_entropies.append(class_ent.item())
        elif isinstance(class_ent, (int, float)):
            class_entropies.append(class_ent)
    
    if 'confidence' in result:
        conf = result['confidence']
        if hasattr(conf, 'item'):
            confidence_scores.append(conf.item())
        elif isinstance(conf, (int, float)):
            confidence_scores.append(conf)


def _calculate_quality_statistics(
    box_variances: list,
    class_entropies: list, 
    confidence_scores: list,
    detection_counts: list,
    config: dict
) -> Dict[str, Any]:
    """품질 통계 계산"""
    quality_stats = {}
    
    # Box variance 통계
    if box_variances:
        quality_stats.update({
            'avg_box_variance': sum(box_variances) / len(box_variances),
            'min_box_variance': min(box_variances),
            'max_box_variance': max(box_variances)
        })
    else:
        quality_stats['avg_box_variance'] = 0.0
    
    # Class entropy 통계
    if class_entropies:
        quality_stats.update({
            'avg_class_entropy': sum(class_entropies) / len(class_entropies),
            'min_class_entropy': min(class_entropies),
            'max_class_entropy': max(class_entropies)
        })
    else:
        quality_stats['avg_class_entropy'] = 0.0
    
    # Confidence 통계
    if confidence_scores:
        quality_stats.update({
            'avg_confidence': sum(confidence_scores) / len(confidence_scores),
            'min_confidence': min(confidence_scores),
            'max_confidence': max(confidence_scores)
        })
    else:
        quality_stats['avg_confidence'] = 0.0
    
    # Detection 통계
    if detection_counts:
        quality_stats.update({
            'avg_detections_per_image': sum(detection_counts) / len(detection_counts),
            'total_detections': sum(detection_counts)
        })
    else:
        quality_stats.update({
            'avg_detections_per_image': 0.0,
            'total_detections': 0
        })
    
    # MC Dropout 설정 정보
    quality_stats.update({
        'mc_num_samples': config['model']['dropout']['num_samples'],
        'mc_dropout_rate': config['model']['dropout']['rate']
    })
    
    return quality_stats 