#!/usr/bin/env python3
"""
MC Dropout num_samples 변화에 따른 불확실성 분포 분석 스크립트

이 스크립트는 다양한 MC sampling 횟수에서의 불확실성 품질을 비교 분석합니다.
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import time
from datetime import datetime
import sys
import warnings
warnings.filterwarnings("ignore")
import cv2
import glob
import random

# 프로젝트 루트를 Python path에 추가
sys.path.append(str(Path(__file__).parent))

from models.yolo_mc import YOLOWithMCDropout
from uncertainty.mc_dropout import MCDropoutDetector

def load_config(config_path: str = "configs/yolo_config.yaml"):
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_real_images(config, num_images: int = 3):
    """실제 COCO 검증 이미지 로드"""
    val_images_path = config['data']['val']['images']
    
    # 이미지 파일 목록 가져오기
    image_patterns = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    
    for pattern in image_patterns:
        image_files.extend(glob.glob(str(Path(val_images_path) / pattern)))
    
    if len(image_files) == 0:
        print(f"❌ 이미지를 찾을 수 없습니다: {val_images_path}")
        return None
    
    # 랜덤하게 이미지 선택
    selected_files = random.sample(image_files, min(num_images, len(image_files)))
    
    images = []
    img_size = config['data']['img_size']
    
    for img_path in selected_files:
        try:
            # 이미지 로드
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # RGB로 변환
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 리사이즈
            img = cv2.resize(img, (img_size, img_size))
            
            # 정규화 (0-1)
            img = img.astype(np.float32) / 255.0
            
            # CHW 형식으로 변환
            img = np.transpose(img, (2, 0, 1))
            
            images.append(img)
            
        except Exception as e:
            print(f"⚠️ 이미지 로드 실패 {img_path}: {e}")
            continue
    
    if len(images) == 0:
        print("❌ 로드된 이미지가 없습니다.")
        return None
    
    # 텐서로 변환
    return torch.tensor(np.array(images), dtype=torch.float32)

def create_dummy_batch(batch_size: int = 4, img_size: int = 640):
    """테스트용 더미 이미지 배치 생성 (백업용)"""
    return torch.randn(batch_size, 3, img_size, img_size)

def analyze_mc_samples_effect(config, output_file: str = "mc_samples_analysis.txt"):
    """
    MC Dropout의 num_samples 변화에 따른 효과 분석
    
    Args:
        config: 설정 딕셔너리
        output_file: 결과 저장 파일명
    """
    
    # 분석할 sample 수 리스트
    sample_counts = [3, 5, 10, 15, 20, 30]
    
    # 결과 저장을 위한 딕셔너리
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'analysis': {}
    }
    
    print("🔬 MC Dropout num_samples 효과 분석 시작...")
    print(f"📊 분석할 sample counts: {sample_counts}")
    
    # 실제 이미지 로드 시도
    test_images = load_real_images(config, num_images=2)
    
    if test_images is None:
        print("⚠️ 실제 이미지 로드 실패. 더미 이미지 사용.")
        test_images = create_dummy_batch(batch_size=2, img_size=config['data']['img_size'])
    
    print(f"📋 테스트 이미지 shape: {test_images.shape}")
    
    # 각 sample count에 대해 분석
    for num_samples in sample_counts:
        print(f"\n🎯 Testing num_samples = {num_samples}")
        
        try:
            # MC Dropout 모델 생성
            model = YOLOWithMCDropout(
                num_classes=config['data']['nc'],
                dropout_rate=config['model']['dropout']['rate']
            )
            model.eval()
            
            # MC Dropout 탐지기 생성
            detector = MCDropoutDetector(
                model=model,
                num_samples=num_samples,  # 샘플 수 변경
                box_std_threshold=config['training']['semi_supervised']['uncertainty']['box_std_threshold'],
                entropy_threshold=config['training']['semi_supervised']['uncertainty']['entropy_threshold']
            )
            
            # 시작 시간 측정
            start_time = time.time()
            
            # MC Dropout 예측 수행
            with torch.no_grad():
                uncertainty_results = detector.predict_with_uncertainty(test_images)
            
            # 실행 시간 측정
            inference_time = time.time() - start_time
            
            # 결과 분석
            analysis_result = analyze_uncertainty_results(
                uncertainty_results, 
                num_samples, 
                inference_time,
                test_images.shape[0]
            )
            
            results['analysis'][num_samples] = analysis_result
            
            print(f"✅ 완료: {inference_time:.3f}초")
            if uncertainty_results:
                print(f"   📊 탐지 개수: {len(uncertainty_results)}")
            
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
    save_results_to_file(results, output_file)
    print(f"\n📁 결과가 {output_file}에 저장되었습니다.")
    
    return results

def analyze_uncertainty_results(uncertainty_results, num_samples, inference_time, batch_size):
    """불확실성 결과 분석"""
    
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
            if 'box_variance' in result:
                box_var = result['box_variance']
                if isinstance(box_var, (int, float)):
                    box_variances.append(box_var)
                elif hasattr(box_var, 'item'):
                    box_variances.append(box_var.item())
            
            if 'class_entropy' in result:
                class_ent = result['class_entropy']
                if isinstance(class_ent, (int, float)):
                    class_entropies.append(class_ent)
                elif hasattr(class_ent, 'item'):
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

def save_results_to_file(results, filename):
    """결과를 포맷된 텍스트 파일로 저장"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MC DROPOUT NUM_SAMPLES 효과 분석 보고서\n")
        f.write("=" * 80 + "\n")
        f.write(f"생성 시간: {results['timestamp']}\n")
        f.write("=" * 80 + "\n\n")
        
        # 요약 테이블
        f.write("📊 요약 테이블\n")
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
                f.write(f"   - 최소값: {stats['box_variance']['min']:.6f}\n")
                f.write(f"   - 최대값: {stats['box_variance']['max']:.6f}\n")
                f.write(f"   - 중간값: {stats['box_variance']['median']:.6f}\n\n")
                
                f.write(f"🎲 Class Entropy (분류 불확실성):\n")
                f.write(f"   - 평균: {stats['class_entropy']['mean']:.6f}\n")
                f.write(f"   - 표준편차: {stats['class_entropy']['std']:.6f}\n")
                f.write(f"   - 최소값: {stats['class_entropy']['min']:.6f}\n")
                f.write(f"   - 최대값: {stats['class_entropy']['max']:.6f}\n")
                f.write(f"   - 중간값: {stats['class_entropy']['median']:.6f}\n\n")
                
                f.write(f"💯 Confidence Scores (신뢰도):\n")
                f.write(f"   - 평균: {stats['confidence_scores']['mean']:.6f}\n")
                f.write(f"   - 표준편차: {stats['confidence_scores']['std']:.6f}\n")
                f.write(f"   - 최소값: {stats['confidence_scores']['min']:.6f}\n")
                f.write(f"   - 최대값: {stats['confidence_scores']['max']:.6f}\n")
                f.write(f"   - 중간값: {stats['confidence_scores']['median']:.6f}\n\n")
            else:
                f.write("   (탐지된 객체 없음)\n\n")
            
            f.write("\n")
        
        # 결론 및 권장사항
        f.write("=" * 80 + "\n")
        f.write("📊 분석 결론 및 권장사항\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("🔍 MC Dropout 샘플 수 효과:\n\n")
        f.write("1. 📈 추론 시간 vs 품질 Trade-off:\n")
        f.write("   - 샘플 수 증가 → 추론 시간 선형 증가\n")
        f.write("   - 샘플 수 증가 → 불확실성 추정 품질 향상\n\n")
        
        f.write("2. 🎯 불확실성 수렴성:\n")
        f.write("   - 낮은 샘플 수: 높은 분산, 불안정한 추정\n")
        f.write("   - 높은 샘플 수: 낮은 분산, 안정적 추정\n\n")
        
        f.write("3. 💡 권장 설정:\n")
        f.write("   - 연구/실험: 15-20 샘플 (높은 품질)\n")
        f.write("   - 실시간 응용: 5-10 샘플 (균형)\n")
        f.write("   - 빠른 테스트: 3-5 샘플 (최소 품질)\n\n")
        
        f.write("4. 🔬 Semi-Supervised Learning 권장:\n")
        f.write("   - Pseudo label 생성: 10-15 샘플\n")
        f.write("   - 신뢰도 기반 필터링에 충분한 품질 확보\n")
        f.write("   - 학습 시간과 품질의 최적 균형점\n\n")
        
        # 성능 비교 차트 (텍스트)
        f.write("📈 성능 비교 차트 (텍스트)\n")
        f.write("-" * 50 + "\n")
        
        # 추론 시간 차트
        f.write("⏱️  추론 시간 (초):\n")
        for num_samples in sorted(results['analysis'].keys()):
            result = results['analysis'][num_samples]
            if 'error' not in result:
                time_val = result['inference_time']
                bar_length = min(int(time_val * 20), 50)  # 스케일링
                bar = "█" * bar_length
                f.write(f"  {num_samples:2d} samples: {time_val:6.3f}s {bar}\n")
        
        f.write("\n")
        
        # 탐지 개수 차트
        f.write("🔍 탐지 개수:\n")
        for num_samples in sorted(results['analysis'].keys()):
            result = results['analysis'][num_samples]
            if 'error' not in result:
                detections = result['uncertainty_stats']['total_detections']
                bar_length = min(detections, 20)  # 최대 20개 막대
                bar = "█" * bar_length
                f.write(f"  {num_samples:2d} samples: {detections:3d} {bar}\n")

def main():
    """메인 실행 함수"""
    print("🚀 MC Dropout num_samples 분석 시작...")
    
    try:
        # 설정 로드
        config = load_config()
        print("✅ 설정 파일 로드 완료")
        
        # 분석 실행
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"mc_samples_analysis_{timestamp}.txt"
        
        results = analyze_mc_samples_effect(config, output_file)
        
        print("\n🎉 분석 완료!")
        print(f"📁 결과 파일: {output_file}")
        
        # 간단한 요약 출력
        print("\n📊 간단 요약:")
        print("-" * 50)
        for num_samples in sorted(results['analysis'].keys()):
            result = results['analysis'][num_samples]
            if 'error' not in result:
                time_val = result['inference_time']
                detections = result['uncertainty_stats']['total_detections']
                print(f"Samples {num_samples:2d}: {time_val:.3f}초, {detections}개 탐지")
            else:
                print(f"Samples {num_samples:2d}: 에러 발생")
        
    except Exception as e:
        print(f"❌ 실행 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 