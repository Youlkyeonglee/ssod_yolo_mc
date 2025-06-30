#!/usr/bin/env python3
"""
학습된 모델을 사용한 MC Dropout 실제 분석

train.py로 학습된 모델을 로드하여 다양한 num_samples에서의 
실제 MC Dropout 효과를 분석합니다.
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import time
from datetime import datetime
import sys
import warnings
import glob
warnings.filterwarnings("ignore")

# 프로젝트 루트를 Python path에 추가
sys.path.append(str(Path(__file__).parent))

from models.yolo_mc import YOLOWithMCDropout
from uncertainty.mc_dropout import MCDropoutDetector
from data.semi_supervised_dataset import SemiSupervisedDataset

def load_config(config_path: str = "configs/yolo_config.yaml"):
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def find_latest_checkpoint(checkpoint_dir: str = "../runs"):
    """가장 최근 학습된 체크포인트 찾기"""
    checkpoint_pattern = f"{checkpoint_dir}/**/best_model.pt"
    checkpoint_files = glob.glob(checkpoint_pattern, recursive=True)
    
    if not checkpoint_files:
        print(f"❌ 체크포인트를 찾을 수 없습니다: {checkpoint_pattern}")
        return None
    
    # 가장 최근 파일 선택
    latest_checkpoint = max(checkpoint_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"✅ 최신 체크포인트 발견: {latest_checkpoint}")
    return latest_checkpoint

def load_trained_model(checkpoint_path: str, config: dict):
    """학습된 모델 로드"""
    print(f"📦 모델 로딩 중: {checkpoint_path}")
    
    # 모델 생성
    model = YOLOWithMCDropout(
        num_classes=config['data']['nc'],
        dropout_rate=config['model']['dropout']['rate']
    )
    
    # 체크포인트 로드
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 상태 딕셔너리 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 모델 상태 로드 완료 (에포크: {checkpoint.get('epoch', 'Unknown')})")
        else:
            # 직접 모델 상태인 경우
            model.load_state_dict(checkpoint)
            print(f"✅ 모델 상태 로드 완료")
            
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return None
    
    model.eval()
    return model

def load_validation_data(config: dict, num_samples: int = 20):
    """검증 데이터 로드"""
    print(f"📋 검증 데이터 로딩 중 (최대 {num_samples}개)...")
    
    try:
        # 검증 데이터셋 생성
        val_dataset = SemiSupervisedDataset(
            images_dir=config['data']['val']['images'],
            labels_dir=config['data']['val']['labels'],
            img_size=config['data']['img_size'],
            is_train=False,
            max_samples=num_samples
        )
        
        if len(val_dataset) == 0:
            print("❌ 검증 데이터가 없습니다.")
            return None
            
        print(f"✅ 검증 데이터 로드 완료: {len(val_dataset)}개 이미지")
        return val_dataset
        
    except Exception as e:
        print(f"❌ 검증 데이터 로딩 실패: {e}")
        return None

def analyze_mc_dropout_on_real_data(model, val_dataset, config, output_file: str):
    """실제 데이터에서 MC Dropout 효과 분석"""
    
    # 분석할 sample 수 리스트
    sample_counts = [3, 5, 10, 15, 20]  # 실제 분석이므로 더 적은 수로 제한
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_info': 'Trained YOLOWithMCDropout',
        'dataset_size': len(val_dataset),
        'analysis': {}
    }
    
    print("🔬 실제 데이터에서 MC Dropout 효과 분석 시작...")
    print(f"📊 분석할 sample counts: {sample_counts}")
    print(f"📋 분석 이미지 수: {len(val_dataset)}")
    
    # 테스트용 이미지 배치 준비
    test_images = []
    for i in range(min(3, len(val_dataset))):  # 3개 이미지만 사용
        image, _ = val_dataset[i]
        test_images.append(image)
    
    if len(test_images) == 0:
        print("❌ 테스트 이미지가 없습니다.")
        return None
    
    test_batch = torch.stack(test_images)
    print(f"📋 테스트 배치 shape: {test_batch.shape}")
    
    # 각 sample count에 대해 분석
    for num_samples in sample_counts:
        print(f"\n🎯 분석 중: num_samples = {num_samples}")
        
        try:
            # MC Dropout 탐지기 생성
            detector = MCDropoutDetector(
                model=model,
                num_samples=num_samples,
                box_std_threshold=config['training']['semi_supervised']['uncertainty']['box_std_threshold'],
                entropy_threshold=config['training']['semi_supervised']['uncertainty']['entropy_threshold']
            )
            
            # 시작 시간 측정
            start_time = time.time()
            
            # MC Dropout 예측 수행
            with torch.no_grad():
                uncertainty_results = detector.predict_with_uncertainty(test_batch)
            
            # 실행 시간 측정
            inference_time = time.time() - start_time
            
            # 결과 분석
            analysis_result = analyze_real_uncertainty_results(
                uncertainty_results, 
                num_samples, 
                inference_time,
                test_batch.shape[0]
            )
            
            results['analysis'][num_samples] = analysis_result
            
            detection_count = len(uncertainty_results) if uncertainty_results else 0
            print(f"✅ 완료: {inference_time:.3f}초, {detection_count}개 탐지")
            
        except Exception as e:
            print(f"❌ 에러 발생 (num_samples={num_samples}): {e}")
            import traceback
            traceback.print_exc()
            results['analysis'][num_samples] = {
                'error': str(e),
                'inference_time': None,
                'uncertainty_stats': None
            }
    
    # 결과를 텍스트 파일로 저장
    save_real_analysis_results(results, output_file)
    print(f"\n📁 결과가 {output_file}에 저장되었습니다.")
    
    return results

def analyze_real_uncertainty_results(uncertainty_results, num_samples, inference_time, batch_size):
    """실제 불확실성 결과 분석"""
    
    # 기본 통계
    total_detections = len(uncertainty_results) if uncertainty_results else 0
    avg_time_per_sample = inference_time / num_samples
    avg_time_per_image = inference_time / batch_size
    
    # 불확실성 통계 계산
    if total_detections > 0:
        box_variances = []
        class_entropies = []
        confidence_scores = []
        
        for result in uncertainty_results:
            # 안전한 값 추출
            if 'box_variance' in result:
                box_var = result['box_variance']
                if isinstance(box_var, (int, float)):
                    box_variances.append(box_var)
                elif hasattr(box_var, 'item'):
                    if box_var.numel() > 0:  # 빈 텐서 체크
                        box_variances.append(box_var.item())
            
            if 'class_entropy' in result:
                class_ent = result['class_entropy']
                if isinstance(class_ent, (int, float)):
                    class_entropies.append(class_ent)
                elif hasattr(class_ent, 'item'):
                    if class_ent.numel() > 0:  # 빈 텐서 체크
                        class_entropies.append(class_ent.item())
            
            if 'confidence' in result:
                conf = result['confidence']
                if isinstance(conf, (int, float)):
                    confidence_scores.append(conf)
                elif hasattr(conf, 'item'):
                    confidence_scores.append(conf.item())
        
        # 통계 계산 (안전 처리)
        def safe_stats(data):
            if len(data) == 0:
                return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
            
            arr = np.array(data)
            return {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'median': float(np.median(arr))
            }
        
        uncertainty_stats = {
            'total_detections': total_detections,
            'box_variance': safe_stats(box_variances),
            'class_entropy': safe_stats(class_entropies),
            'confidence_scores': safe_stats(confidence_scores)
        }
    else:
        uncertainty_stats = {
            'total_detections': 0,
            'box_variance': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0},
            'class_entropy': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0},
            'confidence_scores': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
        }
    
    return {
        'num_samples': num_samples,
        'inference_time': round(inference_time, 4),
        'avg_time_per_sample': round(avg_time_per_sample, 4),
        'avg_time_per_image': round(avg_time_per_image, 4),
        'uncertainty_stats': uncertainty_stats
    }

def save_real_analysis_results(results, filename):
    """실제 분석 결과를 텍스트 파일로 저장"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MC DROPOUT 실제 모델 분석 보고서\n")
        f.write("=" * 80 + "\n")
        f.write(f"생성 시간: {results['timestamp']}\n")
        f.write(f"모델 정보: {results['model_info']}\n")
        f.write(f"분석 데이터 크기: {results['dataset_size']}개 이미지\n")
        f.write("분석 방법: 학습된 모델을 사용한 실제 MC Dropout 분석\n")
        f.write("=" * 80 + "\n\n")
        
        # 요약 테이블
        f.write("📊 실제 성능 분석 테이블\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Samples':<8} {'Time(s)':<10} {'Detections':<12} {'Box Var':<12} {'Entropy':<12} {'Confidence':<12}\n")
        f.write("-" * 80 + "\n")
        
        for num_samples in sorted(results['analysis'].keys()):
            result = results['analysis'][num_samples]
            if 'error' not in result:
                stats = result['uncertainty_stats']
                f.write(f"{num_samples:<8} "
                       f"{result['inference_time']:<10.3f} "
                       f"{stats['total_detections']:<12} "
                       f"{stats['box_variance']['mean']:<12.4f} "
                       f"{stats['class_entropy']['mean']:<12.4f} "
                       f"{stats['confidence_scores']['mean']:<12.4f}\n")
            else:
                f.write(f"{num_samples:<8} ERROR: {result['error']}\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # 상세 분석
        f.write("📋 상세 분석 결과\n")
        f.write("=" * 80 + "\n\n")
        
        for num_samples in sorted(results['analysis'].keys()):
            result = results['analysis'][num_samples]
            
            f.write(f"🎯 NUM_SAMPLES = {num_samples}\n")
            f.write("-" * 40 + "\n")
            
            if 'error' in result:
                f.write(f"❌ 에러: {result['error']}\n\n")
                continue
            
            f.write(f"⏱️  성능 지표:\n")
            f.write(f"   - 총 추론 시간: {result['inference_time']:.4f}초\n")
            f.write(f"   - 샘플당 평균 시간: {result['avg_time_per_sample']:.4f}초\n")
            f.write(f"   - 이미지당 평균 시간: {result['avg_time_per_image']:.4f}초\n\n")
            
            stats = result['uncertainty_stats']
            f.write(f"🔍 탐지 결과:\n")
            f.write(f"   - 총 탐지 개수: {stats['total_detections']}\n\n")
            
            if stats['total_detections'] > 0:
                f.write(f"📦 Box Variance (위치 불확실성):\n")
                f.write(f"   - 평균: {stats['box_variance']['mean']:.6f}\n")
                f.write(f"   - 표준편차: {stats['box_variance']['std']:.6f}\n")
                f.write(f"   - 범위: {stats['box_variance']['min']:.6f} ~ {stats['box_variance']['max']:.6f}\n")
                f.write(f"   - 중간값: {stats['box_variance']['median']:.6f}\n\n")
                
                f.write(f"🎲 Class Entropy (분류 불확실성):\n")
                f.write(f"   - 평균: {stats['class_entropy']['mean']:.6f}\n")
                f.write(f"   - 표준편차: {stats['class_entropy']['std']:.6f}\n")
                f.write(f"   - 범위: {stats['class_entropy']['min']:.6f} ~ {stats['class_entropy']['max']:.6f}\n")
                f.write(f"   - 중간값: {stats['class_entropy']['median']:.6f}\n\n")
                
                f.write(f"💯 Confidence Scores (신뢰도):\n")
                f.write(f"   - 평균: {stats['confidence_scores']['mean']:.6f}\n")
                f.write(f"   - 표준편차: {stats['confidence_scores']['std']:.6f}\n")
                f.write(f"   - 범위: {stats['confidence_scores']['min']:.6f} ~ {stats['confidence_scores']['max']:.6f}\n")
                f.write(f"   - 중간값: {stats['confidence_scores']['median']:.6f}\n\n")
            else:
                f.write("   (탐지된 객체 없음)\n\n")
        
        # 실제 분석 결론
        f.write("=" * 80 + "\n")
        f.write("🎯 실제 모델 분석 결론\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("🔍 실제 모델에서 관찰된 MC Dropout 효과:\n\n")
        
        if any('error' not in results['analysis'][k] for k in results['analysis']):
            successful_results = {k: v for k, v in results['analysis'].items() if 'error' not in v}
            
            if len(successful_results) >= 2:
                min_samples = min(successful_results.keys())
                max_samples = max(successful_results.keys())
                
                min_time = successful_results[min_samples]['inference_time']
                max_time = successful_results[max_samples]['inference_time']
                time_ratio = max_time / min_time if min_time > 0 else 0
                
                f.write(f"1. 📈 추론 시간 증가:\n")
                f.write(f"   - {min_samples} samples: {min_time:.3f}초\n")
                f.write(f"   - {max_samples} samples: {max_time:.3f}초\n")
                f.write(f"   - 증가 비율: {time_ratio:.1f}배\n\n")
                
                min_detection = successful_results[min_samples]['uncertainty_stats']['total_detections']
                max_detection = successful_results[max_samples]['uncertainty_stats']['total_detections']
                
                f.write(f"2. 🎯 탐지 개수 변화:\n")
                f.write(f"   - {min_samples} samples: {min_detection}개\n")
                f.write(f"   - {max_samples} samples: {max_detection}개\n")
                f.write(f"   - 필터링 효과: {'증가' if max_detection > min_detection else '감소' if max_detection < min_detection else '동일'}\n\n")
        
        f.write("💡 실제 사용 권장사항:\n\n")
        f.write("- 이 분석은 실제 학습된 모델을 기반으로 합니다\n")
        f.write("- 실제 데이터에서의 MC Dropout 효과를 확인할 수 있습니다\n")
        f.write("- 학습 품질에 따라 불확실성 추정 성능이 달라질 수 있습니다\n")
        f.write("- 더 많은 데이터와 더 긴 학습으로 개선 가능합니다\n\n")

def main():
    """메인 실행 함수"""
    print("🚀 학습된 모델 MC Dropout 실제 분석 시작...")
    
    try:
        # 설정 로드
        config = load_config()
        print("✅ 설정 파일 로드 완료")
        
        # 최신 체크포인트 찾기
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("❌ 학습된 모델을 찾을 수 없습니다.")
            print("💡 먼저 train.py로 모델을 학습해주세요.")
            return
        
        # 모델 로드
        model = load_trained_model(checkpoint_path, config)
        if model is None:
            return
        
        # 검증 데이터 로드
        val_dataset = load_validation_data(config, num_samples=20)
        if val_dataset is None:
            return
        
        # 분석 실행
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"trained_mc_analysis_{timestamp}.txt"
        
        results = analyze_mc_dropout_on_real_data(model, val_dataset, config, output_file)
        
        if results:
            print("\n🎉 실제 분석 완료!")
            print(f"📁 결과 파일: {output_file}")
            
            # 간단한 요약 출력
            print("\n📊 실제 성능 요약:")
            print("-" * 60)
            print(f"{'Samples':<8} {'Time(s)':<10} {'Detections':<12} {'Box Var':<12}")
            print("-" * 60)
            
            for num_samples in sorted(results['analysis'].keys()):
                result = results['analysis'][num_samples]
                if 'error' not in result:
                    stats = result['uncertainty_stats']
                    print(f"{num_samples:<8} "
                          f"{result['inference_time']:<10.3f} "
                          f"{stats['total_detections']:<12} "
                          f"{stats['box_variance']['mean']:<12.4f}")
                else:
                    print(f"{num_samples:<8} ERROR")
        
    except Exception as e:
        print(f"❌ 실행 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 