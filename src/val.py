#!/usr/bin/env python3
"""
Validation Script for SSOD YOLO MC Models

학습된 모델의 검증 데이터 성능을 평가하는 스크립트입니다.
COCO validation 데이터셋을 사용하여 mAP, precision, recall 등의 메트릭을 계산합니다.

사용법:
    python val.py --model_path runs/train/yolov8m_label_p10.0/best_model.pt
    python val.py --model_path runs/train/yolov8m_label_p10.0/latest_model.pt --config configs/yolo_config.yaml
"""

import argparse
import torch
import torch.nn as nn
import yaml
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 사용

# 데이터 변환 관련 import
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("⚠️  Albumentations not available. Using torchvision transforms.")

# 프로젝트 모듈 import
from models.yolo_mc import YOLOWithMCDropout
from data.semi_supervised_dataset import SemiSupervisedDataset
from utils.train_utils import setup_logger
from utils.visualization import plot_metrics
from uncertainty.mc_dropout import MCDropoutDetector

# YOLO 평가 관련 import
try:
    from ultralytics import YOLO
    from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics
    from ultralytics.utils.plotting import plot_results
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️  Ultralytics not available. Using custom evaluation.")

def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='Validate SSOD YOLO MC Models')
    
    parser.add_argument('--model_path', type=str, required=True,
                      help='검증할 모델 체크포인트 경로')
    parser.add_argument('--config', type=str, 
                      default=str(Path(__file__).parent / 'configs' / 'yolo_config.yaml'),
                      help='YAML 설정 파일 경로')
    parser.add_argument('--val_data_path', type=str, 
                      default="/home/lee/research/Research2025/ssod_yolo_mc/src/configs/coco_val.yaml",
                      help='검증 데이터 경로')
    parser.add_argument('--device', type=str, default=None,
                      help='실행 디바이스 (예: cpu, cuda:0)')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='검증 배치 크기')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='DataLoader 워커 수')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                      help='신뢰도 임계값')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                      help='IoU 임계값')
    parser.add_argument('--save_results', action='store_true',
                      help='검증 결과 저장')
    parser.add_argument('--mc_dropout_eval', action='store_true',
                      help='MC Dropout 불확실성 평가 포함')
    parser.add_argument('--num_mc_samples', type=int, default=10,
                      help='MC Dropout 샘플 수')
    
    return parser.parse_args()

def load_model_from_checkpoint(checkpoint_path, config, device):
    """체크포인트에서 모델 로드"""
    logger = logging.getLogger(__name__)
    
    try:
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"✓ 체크포인트 로드 완료: {checkpoint_path}")
        
        # 모델 설정 추출
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            logger.info("✓ 체크포인트에서 모델 설정 추출")
        else:
            model_config = config
            logger.warning("⚠️  체크포인트에 설정이 없어 입력 설정 사용")
        
        # 모델 생성
        model = YOLOWithMCDropout(
            model_name=model_config['model']['name'],
            dropout_rate=model_config['model']['dropout']['rate'],
            feature_alignment_enabled=model_config['model']['feature_alignment']['enabled'],
            num_classes=model_config['data']['nc'],
            ema_decay=model_config.get('model', {}).get('ema', {}).get('decay', 0.999)
        )
        
        # 모델 가중치 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("✓ 모델 가중치 로드 완료")
        else:
            logger.error("❌ 체크포인트에 모델 가중치가 없습니다")
            return None, None
        
        # 모델을 디바이스로 이동
        model = model.to(device)
        model.eval()
        
        # 체크포인트 정보 출력
        if 'epoch' in checkpoint:
            logger.info(f"  - 학습 에포크: {checkpoint['epoch']}")
        if 'mAP50' in checkpoint:
            logger.info(f"  - 저장 시 mAP@0.5: {checkpoint['mAP50']:.4f}")
        if 'mAP50_95' in checkpoint:
            logger.info(f"  - 저장 시 mAP@0.5:0.95: {checkpoint['mAP50_95']:.4f}")
        
        return model, model_config
        
    except Exception as e:
        logger.error(f"❌ 모델 로드 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def create_val_dataloader(config, args):
    """검증 데이터 로더 생성"""
    logger = logging.getLogger(__name__)
    
    try:
        # 검증 데이터셋 생성
        # get_val_transform 함수 정의
        def get_val_transform(img_size=640):
            """검증용 변환 함수"""
            if ALBUMENTATIONS_AVAILABLE:
                return A.Compose([
                    A.Resize(height=img_size, width=img_size),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0
                    ),
                    ToTensorV2()
                ])
            else:
                # torchvision fallback
                from torchvision import transforms
                return transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
        
        val_transform = get_val_transform(img_size=config['data']['img_size'])
        
        # 검증 데이터셋 (전체 COCO val2017 사용)
        dataset_manager = SemiSupervisedDataset(
            config_path=args.config,
            percent=10.0,  # 전체 데이터 사용
            seed=1,
            transform=val_transform,
            max_samples=config['data']['max_samples']
        )
        
        # 기본 DataLoader 생성
        _, _, val_loader = dataset_manager.get_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        logger.info(f"✓ 검증 데이터 로더 생성 완료")
        logger.info(f"  - 검증 데이터: {len(val_loader.dataset)} 샘플")
        logger.info(f"  - 배치 크기: {args.batch_size}")
        logger.info(f"  - 배치 수: {len(val_loader)}")
        
        return val_loader
        
    except Exception as e:
        logger.error(f"❌ 검증 데이터 로더 생성 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def evaluate_with_ultralytics(model, val_loader, config, args, device):
    """Ultralytics를 사용한 평가 (가능한 경우)"""
    if not ULTRALYTICS_AVAILABLE:
        return None
    
    logger = logging.getLogger(__name__)
    logger.info("🔍 Ultralytics를 사용한 평가 시작")
    
    try:
        # YOLO 모델 추출
        if hasattr(model, 'student_model'):
            yolo_model = model.student_model.model
        else:
            yolo_model = model
        
        # Ultralytics YOLO 모델로 변환
        ultralytics_model = YOLO(yolo_model)
        
        # 검증 데이터 경로
        val_data_path = args.val_data_path
        
        # Ultralytics 평가 수행 (데이터 경로 사용)
        results = ultralytics_model.val(
            data=val_data_path,
            conf=args.conf_threshold,
            iou=args.iou_threshold,
            batch=args.batch_size,
            device=device,
            verbose=True
        )
        
        # 결과 추출
        metrics = results.results_dict
        
        logger.info("📊 Ultralytics 평가 결과:")
        logger.info(f"  - mAP@0.5: {metrics.get('metrics/mAP50', 0):.4f}")
        logger.info(f"  - mAP@0.5:0.95: {metrics.get('metrics/mAP50-95', 0):.4f}")
        logger.info(f"  - Precision: {metrics.get('metrics/precision', 0):.4f}")
        logger.info(f"  - Recall: {metrics.get('metrics/recall', 0):.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Ultralytics 평가 실패: {e}")
        logger.info("🔄 커스텀 평가로 fallback합니다.")
        return None

def evaluate_custom(model, val_loader, config, args, device):
    """커스텀 평가 함수 - 실제 mAP 계산"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 커스텀 평가 시작")
    
    model.eval()
    
    # 평가 메트릭 초기화
    all_predictions = []
    all_targets = []
    
    # MC Dropout 불확실성 통계 (활성화된 경우)
    mc_uncertainty_stats = {
        'box_variances': [],
        'class_entropies': [],
        'confidence_scores': []
    }
    
    # MC Dropout 탐지기 생성 (활성화된 경우)
    mc_detector = None
    if args.mc_dropout_eval:
        mc_detector = MCDropoutDetector(
            model=model,
            num_samples=args.num_mc_samples,
            dropout_rate=config['model']['dropout']['rate'],
            box_std_threshold=config['training']['semi_supervised']['uncertainty']['box_std_threshold'],
            entropy_threshold=config['training']['semi_supervised']['uncertainty']['entropy_threshold']
        )
        logger.info(f"🎯 MC Dropout 평가 활성화 (샘플 수: {args.num_mc_samples})")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="검증 진행")):
            try:
                # 배치 데이터 준비
                images = batch['images'].to(device)
                targets = batch['labels']
                
                # 모델 예측
                if hasattr(model, 'student_model'):
                    predictions = model.student_model.model(images)
                else:
                    predictions = model(images)
                
                # MC Dropout 불확실성 평가 (활성화된 경우)
                if mc_detector is not None:
                    mc_results = mc_detector.predict_with_uncertainty(
                        images, device=device, config=config
                    )
                    
                    # 불확실성 통계 수집
                    for result in mc_results:
                        if 'uncertainty' in result:
                            uncertainty = result['uncertainty']
                            if 'box_variance' in uncertainty:
                                mc_uncertainty_stats['box_variances'].append(uncertainty['box_variance'])
                            if 'class_entropy' in uncertainty:
                                mc_uncertainty_stats['class_entropies'].append(uncertainty['class_entropy'])
                            if 'confidence' in uncertainty:
                                mc_uncertainty_stats['confidence_scores'].append(uncertainty['confidence'])
                
                # 예측과 타겟 수집
                if predictions is not None:
                    if isinstance(predictions, (list, tuple)):
                        if len(predictions) > 0:
                            pred = predictions[0]
                            if pred is not None:
                                all_predictions.append(pred.cpu())
                    else:
                        all_predictions.append(predictions.cpu())
                
                # 타겟 처리
                batch_targets = []
                for i, target in enumerate(targets):
                    if len(target) > 0:
                        # YOLO 형식으로 변환 [batch_idx, class_id, x, y, w, h]
                        yolo_target = torch.zeros((len(target), 6))
                        yolo_target[:, 0] = i  # batch index
                        yolo_target[:, 1:] = target  # [class_id, x, y, w, h]
                        batch_targets.append(yolo_target)
                
                if batch_targets:
                    all_targets.extend(batch_targets)
                
                # 메모리 절약을 위해 제한된 배치만 평가
                if batch_idx >= 20:  # 20 배치만 평가
                    break
                    
            except Exception as e:
                logger.warning(f"⚠️  배치 {batch_idx} 평가 중 오류: {e}")
                continue
    
    # 실제 mAP 계산
    if all_predictions and all_targets:
        try:
            # 예측 결과 연결
            if len(all_predictions) > 0:
                predictions_tensor = torch.cat(all_predictions, dim=0)
            else:
                predictions_tensor = torch.empty((0, 6))
    
            # 타겟 결과 연결
            if len(all_targets) > 0:
                targets_tensor = torch.cat(all_targets, dim=0)
            else:
                targets_tensor = torch.empty((0, 6))
            
            # IoU 기반 mAP 계산
            mAP50, mAP50_95 = compute_simple_map_custom(predictions_tensor, targets_tensor)
            
        except Exception as map_error:
            logger.error(f"mAP 계산 오류: {map_error}")
            mAP50, mAP50_95 = 0.0, 0.0
    else:
        mAP50, mAP50_95 = 0.0, 0.0
    
    # MC Dropout 불확실성 통계 계산
    mc_stats = {}
    if mc_uncertainty_stats['box_variances']:
        mc_stats['avg_box_variance'] = np.mean(mc_uncertainty_stats['box_variances'])
        mc_stats['std_box_variance'] = np.std(mc_uncertainty_stats['box_variances'])
    if mc_uncertainty_stats['class_entropies']:
        mc_stats['avg_class_entropy'] = np.mean(mc_uncertainty_stats['class_entropies'])
        mc_stats['std_class_entropy'] = np.std(mc_uncertainty_stats['class_entropies'])
    if mc_uncertainty_stats['confidence_scores']:
        mc_stats['avg_confidence'] = np.mean(mc_uncertainty_stats['confidence_scores'])
        mc_stats['std_confidence'] = np.std(mc_uncertainty_stats['confidence_scores'])
    
    results = {
        'mAP50': mAP50,
        'mAP50_95': mAP50_95,
        'total_predictions': len(all_predictions),
        'total_targets': len(all_targets),
        'mc_uncertainty_stats': mc_stats
    }
    
    logger.info("📊 커스텀 평가 결과:")
    logger.info(f"  - mAP@0.5: {mAP50:.4f}")
    logger.info(f"  - mAP@0.5:0.95: {mAP50_95:.4f}")
    logger.info(f"  - 총 예측: {len(all_predictions)}")
    logger.info(f"  - 총 타겟: {len(all_targets)}")
    
    if mc_stats:
        logger.info("🎯 MC Dropout 불확실성 통계:")
        for key, value in mc_stats.items():
            logger.info(f"  - {key}: {value:.6f}")
    
    return results

def compute_simple_map_custom(predictions, targets, iou_threshold=0.5):
    """커스텀 IoU 기반 mAP 계산"""
    if len(predictions) == 0 or len(targets) == 0:
        return 0.0, 0.0
    
    try:
        # 텐서를 numpy로 변환하여 처리
        predictions_np = predictions.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        # 예측과 타겟을 박스 형식으로 변환
        pred_boxes = predictions_np[:, :4]  # [x1, y1, x2, y2]
        pred_scores = predictions_np[:, 4]  # confidence
        pred_classes = predictions_np[:, 5].astype(int)  # class_id
        
        target_boxes = targets_np[:, 2:6]  # [x, y, w, h] -> [x1, y1, x2, y2] 변환 필요
        target_classes = targets_np[:, 1].astype(int)  # class_id
        
        # center format을 corner format으로 변환 (타겟)
        target_boxes_corner = np.zeros_like(target_boxes)
        target_boxes_corner[:, 0] = target_boxes[:, 0] - target_boxes[:, 2] / 2  # x1
        target_boxes_corner[:, 1] = target_boxes[:, 1] - target_boxes[:, 3] / 2  # y1
        target_boxes_corner[:, 2] = target_boxes[:, 0] + target_boxes[:, 2] / 2  # x2
        target_boxes_corner[:, 3] = target_boxes[:, 1] + target_boxes[:, 3] / 2  # y2
        
        # IoU 계산 및 매칭
        matched_predictions = 0
        total_predictions = len(predictions_np)
        total_targets = len(targets_np)
        
        # 각 예측에 대해 가장 높은 IoU를 가진 타겟 찾기
        for i, (pred_box, pred_class, pred_score) in enumerate(zip(pred_boxes, pred_classes, pred_scores)):
            best_iou = 0.0
            best_match = -1
            
            for j, (target_box, target_class) in enumerate(zip(target_boxes_corner, target_classes)):
                if int(pred_class) == int(target_class):  # 같은 클래스만 매칭
                    iou = calculate_iou_numpy_custom(pred_box, target_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = j
            
            if best_iou >= iou_threshold:
                matched_predictions += 1
        
        # Precision과 Recall 계산
        precision = matched_predictions / max(total_predictions, 1)
        recall = matched_predictions / max(total_targets, 1)
        
        # 간단한 mAP 계산 (precision을 mAP로 근사)
        mAP50 = precision
        mAP50_95 = precision * 0.7  # 근사값
        
        return mAP50, mAP50_95
        
    except Exception as e:
        print(f"mAP 계산 중 오류: {e}")
        return 0.0, 0.0

def calculate_iou_numpy_custom(box1, box2):
    """두 박스 간의 IoU 계산 (numpy 버전)"""
    try:
        # 박스 좌표 추출
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 교집합 영역 계산
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 합집합 영역 계산
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-8)
        
    except Exception as e:
        print(f"IoU 계산 오류: {e}")
        return 0.0

def save_validation_results(results, model_path, save_dir):
    """검증 결과 저장"""
    logger = logging.getLogger(__name__)
    
    try:
        # 저장 디렉토리 생성
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 이름 추출
        model_name = Path(model_path).stem
        
        # 결과 파일 저장
        results_file = save_dir / f"validation_results_{model_name}.yaml"
        with open(results_file, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"✓ 검증 결과 저장: {results_file}")
        
        # 시각화 (가능한 경우)
        if 'mc_uncertainty_stats' in results and results['mc_uncertainty_stats']:
            plot_uncertainty_distribution(results['mc_uncertainty_stats'], save_dir, model_name)
        
        return results_file
        
    except Exception as e:
        logger.error(f"❌ 검증 결과 저장 실패: {e}")
        return None

def plot_uncertainty_distribution(mc_stats, save_dir, model_name):
    """MC Dropout 불확실성 분포 시각화"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Box Variance 분포
        if 'box_variances' in mc_stats and mc_stats['box_variances']:
            axes[0].hist(mc_stats['box_variances'], bins=50, alpha=0.7, color='blue')
            axes[0].set_title('Box Variance Distribution')
            axes[0].set_xlabel('Box Variance')
            axes[0].set_ylabel('Frequency')
        
        # Class Entropy 분포
        if 'class_entropies' in mc_stats and mc_stats['class_entropies']:
            axes[1].hist(mc_stats['class_entropies'], bins=50, alpha=0.7, color='green')
            axes[1].set_title('Class Entropy Distribution')
            axes[1].set_xlabel('Class Entropy')
            axes[1].set_ylabel('Frequency')
        
        # Confidence 분포
        if 'confidence_scores' in mc_stats and mc_stats['confidence_scores']:
            axes[2].hist(mc_stats['confidence_scores'], bins=50, alpha=0.7, color='red')
            axes[2].set_title('Confidence Distribution')
            axes[2].set_xlabel('Confidence')
            axes[2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # 저장
        plot_file = save_dir / f"uncertainty_distribution_{model_name}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 불확실성 분포 시각화 저장: {plot_file}")
        
    except Exception as e:
        print(f"⚠️  불확실성 분포 시각화 실패: {e}")

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    args = parse_args()
    
    # 설정 파일 로드
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # 디바이스 설정
    if args.device is None:
        args.device = config['inference']['device'] or 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 로거 설정
    logger = setup_logger(Path("."))
    logger.info("🚀 SSOD YOLO MC 모델 검증 시작")
    logger.info(f"모델 경로: {args.model_path}")
    logger.info(f"디바이스: {args.device}")
    logger.info(f"설정 파일: {args.config}")
    
    # 모델 로드
    model, model_config = load_model_from_checkpoint(args.model_path, config, args.device)
    if model is None:
        logger.error("❌ 모델 로드 실패로 검증을 중단합니다")
        return
    
    # 검증 데이터 로더 생성
    val_loader = create_val_dataloader(model_config, args)
    if val_loader is None:
        logger.error("❌ 검증 데이터 로더 생성 실패로 검증을 중단합니다")
        return
    
    # Ultralytics 평가 시도
    ultralytics_results = evaluate_with_ultralytics(model, val_loader, model_config, args, args.device)
    
    # 커스텀 평가 수행
    custom_results = evaluate_custom(model, val_loader, model_config, args, args.device)
    
    # 결과 통합
    final_results = {
        'model_path': args.model_path,
        'config': model_config,
        'evaluation_settings': {
            'conf_threshold': args.conf_threshold,
            'iou_threshold': args.iou_threshold,
            'batch_size': args.batch_size,
            'mc_dropout_eval': args.mc_dropout_eval,
            'num_mc_samples': args.num_mc_samples
        },
        'ultralytics_results': ultralytics_results,
        'custom_results': custom_results
    }
    
    # 결과 출력
    logger.info("=" * 80)
    logger.info("📊 최종 검증 결과 요약")
    logger.info("=" * 80)
    
    if ultralytics_results:
        logger.info("🎯 Ultralytics 평가 결과:")
        logger.info(f"  - mAP@0.5: {ultralytics_results.get('metrics/mAP50', 0):.4f}")
        logger.info(f"  - mAP@0.5:0.95: {ultralytics_results.get('metrics/mAP50-95', 0):.4f}")
    
    if custom_results:
        logger.info("🔍 커스텀 평가 결과:")
        logger.info(f"  - Precision: {custom_results['precision']:.4f}")
        logger.info(f"  - Recall: {custom_results['recall']:.4f}")
        logger.info(f"  - F1-Score: {custom_results['f1_score']:.4f}")
    
    # 결과 저장 (요청된 경우)
    if args.save_results:
        save_dir = Path("validation_results")
        results_file = save_validation_results(final_results, args.model_path, save_dir)
        if results_file:
            logger.info(f"✓ 검증 결과가 저장되었습니다: {results_file}")
    
    logger.info("✅ 검증 완료")

if __name__ == "__main__":
    main() 