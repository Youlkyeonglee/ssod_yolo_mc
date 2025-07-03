#!/usr/bin/env python3
"""
MC Dropout Pseudo Label Distribution Visualization Script

이 스크립트는 MC Dropout의 효과를 확인하기 위해 unlabeled data의 pseudo label 분포를 시각화합니다.
- 그래프1: x축은 클래스명, y축은 갯수
- 그래프2: x축은 cls confidence, y축은 IoU값
- 데이터: labeled, unlabeled, val
"""

from pathlib import Path
import argparse
import torch
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 사용
import matplotlib.font_manager as fm
import numpy as np
from collections import defaultdict, Counter
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한국어 폰트 설정
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass
from models.yolo_mc import YOLOWithMCDropout
from uncertainty.mc_dropout import MCDropoutDetector
from data.semi_supervised_dataset import SemiSupervisedDataset
from utils.train_utils import setup_logger
import torchvision.transforms as transforms
import yaml
from tqdm import tqdm

# COCO 클래스 이름 정의
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def parse_args():
    parser = argparse.ArgumentParser(description='MC Dropout Pseudo Label Distribution Visualization')
    
    # 기본 설정
    parser.add_argument('--config', type=str, 
                      default=str(Path(__file__).parent / 'configs' / 'yolo_config.yaml'),
                      help='YAML 설정 파일 경로')
    
    # 시각화 관련 설정
    parser.add_argument('--save_dir', type=str, default="./visualization_results",
                      help='시각화 결과 저장 디렉토리')
    parser.add_argument('--device', type=str, default="cuda:0",
                      help='실행 디바이스 (예: cpu, cuda:0)')
    parser.add_argument('--max_batches', type=int, default=20,
                      help='분석할 최대 배치 수')
    parser.add_argument('--checkpoint', type=str, default="/media/oem/personal_vol/yklee/ssod_yolo_mc/runs/train/yolov8m_label_p10.0_7/final_model.pth",
                      help='사전 학습된 모델 체크포인트 경로')
    parser.add_argument('--include_uncertainty_plots', action='store_true',
                      help='불확실성 분포 그래프 포함 여부')
    
    args = parser.parse_args()
    
    # YAML 설정 파일 로드
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # config의 값들을 args에 추가 (시각화에 필요한 것만)
    args.model = config['model']['name']
    args.dropout_rate = config['model']['dropout']['rate']
    args.num_samples = config['model']['dropout']['num_samples']
    args.batch_size = config['data']['batch_size']
    args.labeled_ratio = config['data']['labeled_ratio']
    args.seed = config['data']['seed']
    args.max_samples = config['data']['max_samples']
    args.box_std_threshold = config['training']['semi_supervised']['uncertainty']['box_std_threshold']
    args.entropy_threshold = config['training']['semi_supervised']['uncertainty']['entropy_threshold']
    args.conf_threshold = config['training']['semi_supervised']['conf_threshold']
    args.num_workers = config['data']['num_workers']
    
    return args, config

def get_val_transform(img_size: int = 640):
    """Validation transform (no augmentation) - 시각화용"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def calculate_iou(box1, box2):
    """
    두 박스의 IoU 계산
    
    Args:
        box1, box2: [x, y, w, h] 형식의 박스 (normalized coordinates)
    
    Returns:
        IoU 값
    """
    # Convert to [x1, y1, x2, y2] format
    x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    
    x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    # Calculate intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def generate_pseudo_labels_for_visualization(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    detector: MCDropoutDetector,
    device: str,
    logger: logging.Logger,
    config: dict,
    data_type: str = "unlabeled",
    max_batches: int = 20
):
    """
    MC Dropout을 사용하여 pseudo label 생성 및 데이터 수집
    
    Returns:
        pseudo_labels: 생성된 pseudo label 리스트
        gt_labels: ground truth label 리스트 (labeled 데이터의 경우)
    """
    model.eval()  # 평가 모드
    
    pseudo_labels = []
    gt_labels = []
    
    logger.info(f"🔍 {data_type} 데이터에서 pseudo label 생성 시작")
    
    # MC Dropout 설정
    box_std_threshold = config['training']['semi_supervised']['uncertainty']['box_std_threshold']
    entropy_threshold = config['training']['semi_supervised']['uncertainty']['entropy_threshold']
    
    # 진행률 표시
    batch_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Processing {data_type}", 
                                              total=min(len(data_loader), max_batches))):
            if batch_count >= max_batches:
                break
                
            try:
                # 이미지 로드
                images = batch['images'].to(device)
                
                # Ground truth 수집 (labeled 데이터의 경우)
                if data_type == "labeled" and 'labels' in batch:
                    for labels in batch['labels']:
                        if len(labels) > 0:
                            for label in labels:
                                if len(label) >= 5:  # [class_id, x, y, w, h]
                                    try:
                                        class_id = int(label[0].item()) if hasattr(label[0], 'item') else int(label[0])
                                        bbox = label[1:5].cpu().numpy() if hasattr(label[1:5], 'cpu') else label[1:5].numpy()
                                        gt_labels.append({
                                            'class_id': class_id,
                                            'bbox': bbox,
                                            'confidence': 1.0  # GT는 100% 신뢰도
                                        })
                                    except Exception as e:
                                        logger.warning(f"GT 라벨 처리 중 오류: {e}")
                                        continue
                
                # MC Dropout으로 pseudo label 생성
                mc_result = detector.predict_with_uncertainty(
                    images, 
                    device=device, 
                    config=config
                )
                
                if mc_result:
                    for img_idx, result in enumerate(mc_result):
                        if len(result.get('boxes', [])) > 0:
                            boxes = result['boxes']
                            labels = result['labels']
                            scores = result.get('scores', [])
                            
                            for i in range(len(boxes)):
                                pseudo_labels.append({
                                    'class_id': int(labels[i].item()) if hasattr(labels[i], 'item') else int(labels[i]),
                                    'bbox': boxes[i].cpu().numpy() if hasattr(boxes[i], 'cpu') else boxes[i],
                                    'confidence': scores[i].item() if len(scores) > i and hasattr(scores[i], 'item') else 
                                                scores[i] if len(scores) > i else 0.5,
                                    'box_variance': result.get('box_variance', 0.0),
                                    'class_entropy': result.get('class_entropy', 0.0),
                                    'data_type': data_type
                                })
                
                batch_count += 1
                
            except Exception as e:
                logger.warning(f"배치 {batch_idx} 처리 중 오류: {e}")
                continue
    
    logger.info(f"✅ {data_type} 데이터 처리 완료: pseudo_labels={len(pseudo_labels)}, gt_labels={len(gt_labels)}")
    return pseudo_labels, gt_labels

def plot_class_distribution(all_data, save_dir, logger):
    """
    그래프1: x축은 클래스명, y축은 갯수
    """
    logger.info("📊 클래스 분포 그래프 생성 중...")
    
    # 데이터 타입별로 클래스 카운트
    labeled_counts = Counter()
    unlabeled_counts = Counter()
    val_counts = Counter()
    gt_counts = Counter()
    
    for data_type, data_list in all_data.items():
        for item in data_list:
            class_id = item['class_id']
            if class_id < len(COCO_CLASSES):
                class_name = COCO_CLASSES[class_id]
                
                if data_type == 'labeled_pseudo':
                    labeled_counts[class_name] += 1
                elif data_type == 'unlabeled_pseudo':
                    unlabeled_counts[class_name] += 1
                elif data_type == 'val_pseudo':
                    val_counts[class_name] += 1
                elif data_type == 'gt':
                    gt_counts[class_name] += 1
    
    # 전체 클래스 이름 수집 (최소 1개 이상 검출된 클래스만)
    all_classes = set()
    all_classes.update(labeled_counts.keys())
    all_classes.update(unlabeled_counts.keys())
    all_classes.update(val_counts.keys())
    all_classes.update(gt_counts.keys())
    all_classes = sorted(list(all_classes))
    
    if not all_classes:
        logger.warning("검출된 클래스가 없습니다.")
        return
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(15, 8))
    
    x_pos = np.arange(len(all_classes))
    width = 0.2
    
    # 각 데이터 타입별 막대 그래프
    labeled_values = [labeled_counts.get(cls, 0) for cls in all_classes]
    unlabeled_values = [unlabeled_counts.get(cls, 0) for cls in all_classes]
    val_values = [val_counts.get(cls, 0) for cls in all_classes]
    gt_values = [gt_counts.get(cls, 0) for cls in all_classes]
    
    ax.bar(x_pos - 1.5*width, labeled_values, width, label='Labeled Pseudo', alpha=0.8, color='skyblue')
    ax.bar(x_pos - 0.5*width, unlabeled_values, width, label='Unlabeled Pseudo', alpha=0.8, color='lightcoral')
    ax.bar(x_pos + 0.5*width, val_values, width, label='Val Pseudo', alpha=0.8, color='lightgreen')
    ax.bar(x_pos + 1.5*width, gt_values, width, label='Ground Truth', alpha=0.8, color='gold')
    
    ax.set_xlabel('클래스명', fontsize=12)
    ax.set_ylabel('검출 갯수', fontsize=12)
    ax.set_title('MC Dropout Pseudo Label vs Ground Truth 클래스 분포', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    save_path = save_dir / 'class_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ 클래스 분포 그래프 저장: {save_path}")

def plot_confidence_iou_scatter(all_data, save_dir, logger):
    """
    그래프2: x축은 cls confidence, y축은 IoU값
    """
    logger.info("📊 Confidence-IoU 분산도 그래프 생성 중...")
    
    # GT와 pseudo label 간 IoU 계산을 위한 데이터 준비
    gt_data = all_data.get('gt', [])
    pseudo_data = []
    
    # 모든 pseudo label 수집
    for data_type in ['labeled_pseudo', 'unlabeled_pseudo', 'val_pseudo']:
        if data_type in all_data:
            for item in all_data[data_type]:
                item['source'] = data_type
                pseudo_data.append(item)
    
    if not gt_data or not pseudo_data:
        logger.warning("GT 또는 pseudo label 데이터가 부족하여 IoU 계산을 건너뜁니다.")
        return
    
    # Confidence-IoU 데이터 수집
    conf_iou_data = []
    
    for pseudo in pseudo_data:
        best_iou = 0.0
        
        # 같은 클래스의 GT와 IoU 계산
        for gt in gt_data:
            if gt['class_id'] == pseudo['class_id']:
                try:
                    iou = calculate_iou(pseudo['bbox'], gt['bbox'])
                    best_iou = max(best_iou, iou)
                except:
                    continue
        
        conf_iou_data.append({
            'confidence': pseudo['confidence'],
            'iou': best_iou,
            'source': pseudo['source']
        })
    
    if not conf_iou_data:
        logger.warning("유효한 Confidence-IoU 데이터가 없습니다.")
        return
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 소스별로 색상 분리
    source_colors = {
        'labeled_pseudo': 'skyblue',
        'unlabeled_pseudo': 'lightcoral', 
        'val_pseudo': 'lightgreen'
    }
    
    source_labels = {
        'labeled_pseudo': 'Labeled Pseudo',
        'unlabeled_pseudo': 'Unlabeled Pseudo',
        'val_pseudo': 'Val Pseudo'
    }
    
    for source in source_colors.keys():
        source_data = [item for item in conf_iou_data if item['source'] == source]
        if source_data:
            confidences = [item['confidence'] for item in source_data]
            ious = [item['iou'] for item in source_data]
            
            ax.scatter(confidences, ious, 
                      c=source_colors[source], 
                      label=source_labels[source],
                      alpha=0.6, s=20)
    
    ax.set_xlabel('Classification Confidence', fontsize=12)
    ax.set_ylabel('IoU with Ground Truth', fontsize=12)
    ax.set_title('MC Dropout Pseudo Label: Confidence vs IoU', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 대각선 참조선 추가 (이상적인 경우)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Perfect Correlation')
    
    plt.tight_layout()
    
    # 저장
    save_path = save_dir / 'confidence_iou_scatter.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Confidence-IoU 분산도 저장: {save_path}")
    
    # 통계 정보 로깅
    avg_confidence = np.mean([item['confidence'] for item in conf_iou_data])
    avg_iou = np.mean([item['iou'] for item in conf_iou_data])
    
    logger.info(f"📈 평균 Confidence: {avg_confidence:.3f}")
    logger.info(f"📈 평균 IoU: {avg_iou:.3f}")

def plot_uncertainty_distributions(all_data, save_dir, logger):
    """
    그래프3: MC Dropout 불확실성 분포 히스토그램
    """
    logger.info("📊 불확실성 분포 히스토그램 생성 중...")
    
    # 불확실성 데이터 수집
    box_variances = []
    class_entropies = []
    data_sources = []
    
    for data_type in ['labeled_pseudo', 'unlabeled_pseudo', 'val_pseudo']:
        if data_type in all_data:
            for item in all_data[data_type]:
                if 'box_variance' in item and 'class_entropy' in item:
                    box_variances.append(item['box_variance'])
                    class_entropies.append(item['class_entropy'])
                    data_sources.append(data_type)
    
    if not box_variances or not class_entropies:
        logger.warning("불확실성 데이터가 부족하여 히스토그램을 건너뜁니다.")
        return
    
    # 그래프 생성 (2x2 서브플롯)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Box Variance 히스토그램
    ax1.hist(box_variances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Box Variance')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Box Variance Distribution (MC Dropout)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Class Entropy 히스토그램  
    ax2.hist(class_entropies, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Class Entropy')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Class Entropy Distribution (MC Dropout)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Box Variance vs Class Entropy 산점도
    ax3.scatter(box_variances, class_entropies, alpha=0.6, s=20, c='green')
    ax3.set_xlabel('Box Variance')
    ax3.set_ylabel('Class Entropy')
    ax3.set_title('Box Variance vs Class Entropy')
    ax3.grid(True, alpha=0.3)
    
    # 4. 데이터 소스별 불확실성 박스플롯
    source_data = defaultdict(list)
    for i, source in enumerate(data_sources):
        source_data[source].append(box_variances[i])
    
    if len(source_data) > 1:
        box_data = [source_data[key] for key in source_data.keys()]
        box_labels = list(source_data.keys())
        ax4.boxplot(box_data, labels=box_labels)
        ax4.set_ylabel('Box Variance')
        ax4.set_title('Box Variance by Data Source')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor boxplot', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Box Variance by Data Source')
    
    plt.tight_layout()
    
    # 저장
    save_path = save_dir / 'uncertainty_distributions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ 불확실성 분포 히스토그램 저장: {save_path}")
    
    # 통계 정보 로깅
    logger.info(f"📈 Box Variance - 평균: {np.mean(box_variances):.4f}, 표준편차: {np.std(box_variances):.4f}")
    logger.info(f"📈 Class Entropy - 평균: {np.mean(class_entropies):.4f}, 표준편차: {np.std(class_entropies):.4f}")

def plot_confidence_calibration(all_data, save_dir, logger):
    """
    그래프4: 신뢰도 캘리브레이션 곡선
    """
    logger.info("📊 신뢰도 캘리브레이션 곡선 생성 중...")
    
    # GT와 pseudo label 간 매칭 데이터 수집
    gt_data = all_data.get('gt', [])
    calibration_data = []
    
    for data_type in ['labeled_pseudo', 'unlabeled_pseudo', 'val_pseudo']:
        if data_type in all_data:
            for pseudo in all_data[data_type]:
                best_iou = 0.0
                is_correct = False
                
                # 같은 클래스의 GT와 IoU 계산
                for gt in gt_data:
                    if gt['class_id'] == pseudo['class_id']:
                        try:
                            iou = calculate_iou(pseudo['bbox'], gt['bbox'])
                            if iou > best_iou:
                                best_iou = iou
                                is_correct = iou > 0.5  # IoU > 0.5를 정답으로 간주
                        except:
                            continue
                
                calibration_data.append({
                    'confidence': pseudo['confidence'],
                    'correct': is_correct,
                    'source': data_type
                })
    
    if len(calibration_data) < 10:
        logger.warning("캘리브레이션 분석을 위한 데이터가 부족합니다.")
        return
    
    # 신뢰도 구간별 정확도 계산
    confidence_bins = np.linspace(0, 1, 11)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(len(confidence_bins) - 1):
        bin_start, bin_end = confidence_bins[i], confidence_bins[i + 1]
        bin_data = [d for d in calibration_data 
                   if bin_start <= d['confidence'] < bin_end]
        
        if bin_data:
            bin_accuracy = sum(d['correct'] for d in bin_data) / len(bin_data)
            bin_confidence = np.mean([d['confidence'] for d in bin_data])
            
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(len(bin_data))
        else:
            bin_accuracies.append(0)
            bin_confidences.append((bin_start + bin_end) / 2)
            bin_counts.append(0)
    
    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 캘리브레이션 곡선
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    ax1.plot(bin_confidences, bin_accuracies, 'ro-', linewidth=2, markersize=8, 
             label='MC Dropout Predictions')
    
    # 각 점에 샘플 수 표시
    for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
        if count > 0:
            ax1.annotate(f'{count}', (conf, acc), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Reliability Diagram (Calibration Curve)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # 2. 신뢰도 히스토그램
    confidences = [d['confidence'] for d in calibration_data]
    ax2.hist(confidences, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Number of Predictions')
    ax2.set_title('Confidence Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    save_path = save_dir / 'confidence_calibration.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ 신뢰도 캘리브레이션 곡선 저장: {save_path}")
    
    # Expected Calibration Error (ECE) 계산
    ece = sum(abs(acc - conf) * count for acc, conf, count in 
              zip(bin_accuracies, bin_confidences, bin_counts)) / sum(bin_counts)
    logger.info(f"📈 Expected Calibration Error (ECE): {ece:.4f}")

def create_summary_statistics(all_data, save_dir, logger):
    """
    전체 통계 정보를 텍스트 파일로 저장
    """
    logger.info("📋 요약 통계 생성 중...")
    
    summary_file = save_dir / 'summary_statistics.txt'
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MC DROPOUT PSEUDO LABEL 분포 분석 요약 보고서\n")
        f.write("=" * 80 + "\n\n")
        
        # 전체 통계
        f.write("📊 전체 데이터 통계\n")
        f.write("-" * 50 + "\n")
        
        for data_type, data_list in all_data.items():
            f.write(f"{data_type}: {len(data_list)}개\n")
        
        f.write("\n")
        
        # 클래스별 통계
        f.write("📈 클래스별 검출 통계\n")
        f.write("-" * 50 + "\n")
        
        class_stats = defaultdict(lambda: defaultdict(int))
        for data_type, data_list in all_data.items():
            for item in data_list:
                class_id = item['class_id']
                if class_id < len(COCO_CLASSES):
                    class_name = COCO_CLASSES[class_id]
                    class_stats[class_name][data_type] += 1
        
        for class_name in sorted(class_stats.keys()):
            f.write(f"\n{class_name}:\n")
            for data_type, count in class_stats[class_name].items():
                f.write(f"  - {data_type}: {count}개\n")
        
        # Confidence 통계 (pseudo label만)
        f.write("\n📊 Confidence 통계 (Pseudo Labels)\n")
        f.write("-" * 50 + "\n")
        
        for data_type in ['labeled_pseudo', 'unlabeled_pseudo', 'val_pseudo']:
            if data_type in all_data:
                confidences = [item['confidence'] for item in all_data[data_type]]
                if confidences:
                    f.write(f"\n{data_type}:\n")
                    f.write(f"  - 평균: {np.mean(confidences):.3f}\n")
                    f.write(f"  - 표준편차: {np.std(confidences):.3f}\n")
                    f.write(f"  - 최소값: {np.min(confidences):.3f}\n")
                    f.write(f"  - 최대값: {np.max(confidences):.3f}\n")
                    f.write(f"  - 중앙값: {np.median(confidences):.3f}\n")
        
        # 불확실성 통계 (MC Dropout)
        f.write("\n📊 불확실성 통계 (MC Dropout)\n")
        f.write("-" * 50 + "\n")
        
        all_box_variances = []
        all_class_entropies = []
        
        for data_type in ['labeled_pseudo', 'unlabeled_pseudo', 'val_pseudo']:
            if data_type in all_data:
                box_variances = [item.get('box_variance', 0) for item in all_data[data_type]]
                class_entropies = [item.get('class_entropy', 0) for item in all_data[data_type]]
                
                if box_variances and any(v > 0 for v in box_variances):
                    f.write(f"\n{data_type} - Box Variance:\n")
                    f.write(f"  - 평균: {np.mean(box_variances):.4f}\n")
                    f.write(f"  - 표준편차: {np.std(box_variances):.4f}\n")
                    f.write(f"  - 최소값: {np.min(box_variances):.4f}\n")
                    f.write(f"  - 최대값: {np.max(box_variances):.4f}\n")
                    
                if class_entropies and any(e > 0 for e in class_entropies):
                    f.write(f"\n{data_type} - Class Entropy:\n")
                    f.write(f"  - 평균: {np.mean(class_entropies):.4f}\n")
                    f.write(f"  - 표준편차: {np.std(class_entropies):.4f}\n")
                    f.write(f"  - 최소값: {np.min(class_entropies):.4f}\n")
                    f.write(f"  - 최대값: {np.max(class_entropies):.4f}\n")
                
                all_box_variances.extend(box_variances)
                all_class_entropies.extend(class_entropies)
        
        # 전체 불확실성 요약
        if all_box_variances and any(v > 0 for v in all_box_variances):
            f.write(f"\n전체 Box Variance 요약:\n")
            f.write(f"  - 평균: {np.mean(all_box_variances):.4f}\n")
            f.write(f"  - 중앙값: {np.median(all_box_variances):.4f}\n")
            f.write(f"  - 90% 분위수: {np.percentile(all_box_variances, 90):.4f}\n")
            
        if all_class_entropies and any(e > 0 for e in all_class_entropies):
            f.write(f"\n전체 Class Entropy 요약:\n")
            f.write(f"  - 평균: {np.mean(all_class_entropies):.4f}\n")
            f.write(f"  - 중앙값: {np.median(all_class_entropies):.4f}\n")
            f.write(f"  - 90% 분위수: {np.percentile(all_class_entropies, 90):.4f}\n")
        
        # 분석 권장사항
        f.write("\n🔍 분석 권장사항\n")
        f.write("-" * 50 + "\n")
        f.write("1. 클래스 분포의 불균형 정도를 확인하세요.\n")
        f.write("2. Confidence-IoU 분산도에서 캘리브레이션 품질을 평가하세요.\n")
        f.write("3. Box variance가 높은 예측은 불확실성이 크므로 필터링을 고려하세요.\n")
        f.write("4. Class entropy가 높은 예측은 클래스 분류에 불확실성이 있습니다.\n")
        f.write("5. Unlabeled 데이터의 pseudo label 품질을 labeled 데이터와 비교해보세요.\n")
    
    logger.info(f"✅ 요약 통계 저장: {summary_file}")

def main():
    # 명령행 인자와 설정 파싱
    args, config = parse_args()
    
    # 저장 디렉토리 생성
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 로거 설정
    logger = setup_logger(save_dir)
    logger.info("🎨 MC Dropout Pseudo Label 분포 시각화 시작")
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"사용 디바이스: {device}")
    
    # MC Dropout 모델 생성
    try:
        model = YOLOWithMCDropout(
            model_name=args.model,
            dropout_rate=args.dropout_rate,
            feature_alignment_enabled=False,  # 시각화에서는 불필요
            num_classes=config['data']['nc']
        ).to(device)
        
        # 체크포인트 로드 (있는 경우)
        if args.checkpoint and Path(args.checkpoint).exists():
            logger.info(f"체크포인트 로딩: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            
            # 모델 상태 딕셔너리 로드
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            logger.info("✅ 체크포인트 로드 완료")
        else:
            logger.info("사전 학습된 체크포인트 없이 기본 모델 사용")
            
    except Exception as e:
        logger.error(f"모델 생성 실패: {e}")
        raise
    
    # MC Dropout 탐지기 생성
    try:
        detector = MCDropoutDetector(
            model=model,
            num_samples=args.num_samples,
            dropout_rate=args.dropout_rate,
            box_std_threshold=args.box_std_threshold,
            entropy_threshold=args.entropy_threshold
        )
        logger.info("✅ MC Dropout 탐지기 생성 완료")
    except Exception as e:
        logger.error(f"MC Dropout 탐지기 생성 실패: {e}")
        raise
    
    # 데이터셋 로드
    logger.info("📂 데이터셋 로딩 중...")
    transform = get_val_transform(img_size=config['data']['img_size'])
    
    dataset_manager = SemiSupervisedDataset(
        config_path=args.config,
        percent=args.labeled_ratio,
        seed=args.seed,
        transform=transform,
        max_samples=args.max_samples
    )
    
    # 데이터 로더 생성
    labeled_loader, unlabeled_loader, val_loader = dataset_manager.get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    logger.info(f"✅ 데이터셋 로드 완료")
    logger.info(f"  - Labeled: {len(labeled_loader.dataset)} 샘플")
    logger.info(f"  - Unlabeled: {len(unlabeled_loader.dataset)} 샘플") 
    logger.info(f"  - Validation: {len(val_loader.dataset)} 샘플")
    
    # 각 데이터셋에서 pseudo label 생성
    all_data = {}
    
    # 1. Labeled 데이터 처리 (pseudo label + ground truth)
    logger.info("🔍 Labeled 데이터 처리 중...")
    labeled_pseudo, labeled_gt = generate_pseudo_labels_for_visualization(
        model=model,
        data_loader=labeled_loader,
        detector=detector,
        device=device,
        logger=logger,
        config=config,
        data_type="labeled",
        max_batches=args.max_batches
    )
    all_data['labeled_pseudo'] = labeled_pseudo
    all_data['gt'] = labeled_gt
    
    # 2. Unlabeled 데이터 처리 (pseudo label만)
    logger.info("🔍 Unlabeled 데이터 처리 중...")
    unlabeled_pseudo, _ = generate_pseudo_labels_for_visualization(
        model=model,
        data_loader=unlabeled_loader,
        detector=detector,
        device=device,
        logger=logger,
        config=config,
        data_type="unlabeled",
        max_batches=args.max_batches
    )
    all_data['unlabeled_pseudo'] = unlabeled_pseudo
    
    # 3. Validation 데이터 처리 (pseudo label만)
    logger.info("🔍 Validation 데이터 처리 중...")
    val_pseudo, _ = generate_pseudo_labels_for_visualization(
        model=model,
        data_loader=val_loader,
        detector=detector,
        device=device,
        logger=logger,
        config=config,
        data_type="val",
        max_batches=args.max_batches
    )
    all_data['val_pseudo'] = val_pseudo
    
    # 시각화 생성
    logger.info("🎨 시각화 생성 중...")
    
    try:
        # 그래프1: 클래스 분포
        plot_class_distribution(all_data, save_dir, logger)
        
        # 그래프2: Confidence-IoU 분산도
        plot_confidence_iou_scatter(all_data, save_dir, logger)
        
        # 그래프3,4: 불확실성 분포 및 캘리브레이션 (옵션)
        if args.include_uncertainty_plots:
            plot_uncertainty_distributions(all_data, save_dir, logger)
            plot_confidence_calibration(all_data, save_dir, logger)
        
        # 요약 통계
        create_summary_statistics(all_data, save_dir, logger)
        
        # 최종 결과 요약
        total_labeled = len(all_data.get('labeled_pseudo', []))
        total_unlabeled = len(all_data.get('unlabeled_pseudo', []))
        total_val = len(all_data.get('val_pseudo', []))
        total_gt = len(all_data.get('gt', []))
        
        logger.info("=" * 60)
        logger.info("📊 최종 분석 결과 요약")
        logger.info("=" * 60)
        logger.info(f"📈 Labeled Pseudo Labels: {total_labeled}개")
        logger.info(f"📈 Unlabeled Pseudo Labels: {total_unlabeled}개") 
        logger.info(f"📈 Validation Pseudo Labels: {total_val}개")
        logger.info(f"📈 Ground Truth Labels: {total_gt}개")
        logger.info(f"📈 총 Pseudo Labels: {total_labeled + total_unlabeled + total_val}개")
        
        # 생성된 파일 목록
        generated_files = []
        for file_path in save_dir.glob('*.png'):
            generated_files.append(file_path.name)
        for file_path in save_dir.glob('*.txt'):
            generated_files.append(file_path.name)
            
        logger.info(f"📁 생성된 파일: {', '.join(generated_files)}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"시각화 생성 중 오류 발생: {e}")
        raise
    
    logger.info("🎉 MC Dropout Pseudo Label 분포 시각화 완료!")
    logger.info(f"📁 결과 저장 위치: {save_dir}")

if __name__ == "__main__":
    main() 