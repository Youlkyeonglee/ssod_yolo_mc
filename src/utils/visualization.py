import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from typing import List, Tuple, Dict, Optional
import os

def visualize_batch(
    images: torch.Tensor,
    targets: List[torch.Tensor],
    predictions: List[torch.Tensor],
    class_names: List[str],
    save_path: Path,
    max_images: int = 16,
    img_paths: Optional[List[str]] = None
) -> None:
    """배치 이미지와 예측 결과를 시각화하여 저장
    
    Args:
        images: 배치 이미지 텐서 (B, C, H, W), 정규화된 값 (0-1)
        targets: 정답 박스 리스트 [(N, 5), ...], 5 = (class_id, x_center, y_center, width, height) - 정규화됨
        predictions: 예측 박스 리스트 [(M, 6), ...], 6 = (x1, y1, x2, y2, conf, class_id) - 픽셀 좌표
        class_names: 클래스 이름 리스트
        save_path: 저장 경로
        max_images: 시각화할 최대 이미지 수
        img_paths: 이미지 파일 경로 리스트 (제목 표시용, 선택사항)
    """
    # 배치 크기 제한
    batch_size = min(len(images), max_images)
    
    # 서브플롯 크기 계산
    n_cols = min(4, batch_size)
    n_rows = (batch_size - 1) // n_cols + 1
    
    plt.figure(figsize=(n_cols * 6, n_rows * 6))
    
    for i in range(batch_size):
        # 이미지 변환 (텐서 -> numpy)
        img = images[i].permute(1, 2, 0).cpu().numpy()
        
        # 이미지가 이미 정규화 해제된 상태인지 확인
        if img.max() > 1.0:  # ImageNet 정규화가 적용된 경우
            # ImageNet 정규화 해제
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
        
        # 0-255 범위로 변환
        img = (img * 255).astype(np.uint8)
        
        # 이미지 크기 (transform 후 크기)
        img_h, img_w = img.shape[:2]
        
        # 디버깅: 첫 번째 이미지에서 크기 정보 출력
        if i == 0:
            print(f"Display image size: {img_w}x{img_h}")
        
        # 서브플롯 생성
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(img)
        
        # 정답 박스 그리기 (녹색)
        if i < len(targets) and targets[i] is not None and len(targets[i]) > 0:
            target_boxes = targets[i].cpu().numpy() if torch.is_tensor(targets[i]) else targets[i]
            
            for box in target_boxes:
                if len(box) >= 5:  # class_id, x_center, y_center, width, height
                    class_id = box[0]
                    
                    # class_id 안전 변환 (배열/텐서 -> 스칼라)
                    if hasattr(class_id, '__len__') and len(class_id) > 0:
                        class_id = class_id[0]
                    elif hasattr(class_id, 'item'):  # PyTorch tensor
                        class_id = class_id.item()
                    class_id = int(class_id)
                    
                    # 클래스 ID 유효성 검증
                    if 0 <= class_id < len(class_names):
                        # center + width/height 형식 -> 픽셀 좌표 변환
                        x_center, y_center, width, height = box[1:5]
                        
                        # 픽셀 좌표 계산 (center -> top-left corner)
                        x1 = (x_center - width / 2) * img_w
                        y1 = (y_center - height / 2) * img_h
                        x2 = (x_center + width / 2) * img_w
                        y2 = (y_center + height / 2) * img_h
                        
                        # 경계 체크
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img_w-1, x2), min(img_h-1, y2)
                        
                        # 박스 그리기
                        rect = plt.Rectangle(
                            (x1, y1), x2-x1, y2-y1,
                            fill=False, color='lime', linewidth=3, alpha=0.8
                        )
                        plt.gca().add_patch(rect)
                        
                        # 클래스 이름 표시
                        plt.text(
                            x1, y1-8,
                            f'GT: {class_names[class_id]}',
                            color='lime',
                            fontsize=10,
                            fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.5)
                        )
        
        # 예측 박스 그리기 (빨간색)
        if i < len(predictions) and predictions[i] is not None:
            try:
                # predictions[i] 안전 처리
                pred_data = predictions[i]
                
                # 디버깅: predictions 데이터 구조 확인 (임시)
                # print(f"DEBUG: Image {i}, pred_data type: {type(pred_data)}")
                # if hasattr(pred_data, 'shape'):
                #     print(f"DEBUG: pred_data shape: {pred_data.shape}")
                # elif hasattr(pred_data, '__len__'):
                #     print(f"DEBUG: pred_data length: {len(pred_data)}")
                #     if len(pred_data) > 0 and hasattr(pred_data[0], '__len__'):
                #         print(f"DEBUG: first element length: {len(pred_data[0])}")
                #         if len(pred_data[0]) > 0:
                #             print(f"DEBUG: first box sample: {pred_data[0][:1] if len(pred_data[0]) > 0 else 'empty'}")
                
                # 다양한 타입의 prediction 데이터 처리
                if torch.is_tensor(pred_data):
                    pred_boxes = pred_data.cpu().numpy()
                elif hasattr(pred_data, 'cpu'):  # GPU tensor with custom class
                    pred_boxes = pred_data.cpu().numpy()
                elif isinstance(pred_data, (list, tuple)):
                    pred_boxes = pred_data
                elif hasattr(pred_data, '__iter__'):
                    pred_boxes = list(pred_data)
                else:
                    pred_boxes = [pred_data] if pred_data is not None else []
                
                # 빈 데이터 체크
                if len(pred_boxes) == 0:
                    continue
                
                # 각 박스 처리
                for box_idx, box in enumerate(pred_boxes):
                    try:
                        # box가 tensor인 경우 numpy로 변환
                        if torch.is_tensor(box):
                            box = box.cpu().numpy()
                        
                        # box 길이 체크
                        if hasattr(box, '__len__') and len(box) >= 6:
                            x1, y1, x2, y2, conf, class_id = box[:6]
                        elif hasattr(box, '__len__') and len(box) >= 5:
                            x1, y1, x2, y2, conf = box[:5]
                            class_id = 0  # 기본 클래스
                        else:
                            continue  # 유효하지 않은 박스
                        
                        # conf 안전 변환 (다양한 타입 -> 스칼라)
                        try:
                            if torch.is_tensor(conf):
                                # PyTorch tensor인 경우
                                if conf.numel() == 1:
                                    conf = conf.item()
                                elif conf.numel() > 1:
                                    conf = conf.flatten()[0].item()
                                else:
                                    conf = 0.0  # 빈 텐서인 경우
                            elif hasattr(conf, '__len__') and len(conf) > 0:
                                # numpy array나 list인 경우
                                conf = float(conf[0])
                            elif hasattr(conf, '__iter__'):
                                # 기타 iterable인 경우
                                conf = float(next(iter(conf)))
                            else:
                                # 이미 스칼라인 경우
                                conf = float(conf)
                        except (ValueError, TypeError, IndexError):
                            conf = 0.0  # 변환 실패 시 기본값
                        
                        # class_id 안전 변환 (다양한 타입 -> 스칼라)
                        try:
                            if torch.is_tensor(class_id):
                                # PyTorch tensor인 경우
                                if class_id.numel() == 1:
                                    class_id = class_id.item()
                                elif class_id.numel() > 1:
                                    class_id = class_id.flatten()[0].item()
                                else:
                                    class_id = 0  # 빈 텐서인 경우
                            elif hasattr(class_id, '__len__') and len(class_id) > 0:
                                # numpy array나 list인 경우
                                class_id = int(class_id[0])
                            elif hasattr(class_id, '__iter__'):
                                # 기타 iterable인 경우
                                class_id = int(next(iter(class_id)))
                            else:
                                # 이미 스칼라인 경우
                                class_id = int(class_id)
                        except (ValueError, TypeError, IndexError):
                            class_id = 0  # 변환 실패 시 기본값
                        
                        # 좌표값 안전 변환
                        try:
                            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                        except (ValueError, TypeError):
                            continue  # 좌표 변환 실패 시 건너뛰기
                        
                        # 클래스 ID 유효성 검증
                        if 0 <= class_id < len(class_names) and conf > 0.3:  # confidence threshold
                            # 경계 체크
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(img_w-1, x2), min(img_h-1, y2)
                            
                            # 박스 크기 체크 (너무 작거나 잘못된 박스 제외)
                            if x2 > x1 and y2 > y1:
                                # 박스 그리기
                                rect = plt.Rectangle(
                                    (x1, y1), x2-x1, y2-y1,
                                    fill=False, color='red', linewidth=2, alpha=0.8
                                )
                                plt.gca().add_patch(rect)
                                
                                # 클래스 이름과 confidence 표시
                                plt.text(
                                    x1, y2+15,
                                    f'Pred: {class_names[class_id]} {conf:.2f}',
                                    color='red',
                                    fontsize=9,
                                    fontweight='bold',
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.5)
                                )
                    
                    except Exception as box_error:
                        print(f"Warning: Error processing box {box_idx} in image {i}: {box_error}")
                        continue
                        
            except Exception as pred_error:
                print(f"Warning: Error processing predictions for image {i}: {pred_error}")
                continue
        
        # 이미지 제목 설정
        img_name = img_paths[i].split('/')[-1] if img_paths and i < len(img_paths) else f'Image {i+1}'
        plt.title(img_name, fontsize=12, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {save_path}")

def visualize_gt_data(
    data_loader: torch.utils.data.DataLoader,
    class_names: List[str],
    save_dir: Path,
    data_type: str = "train",
    max_batches: int = 3,
    max_images_per_batch: int = 8
) -> None:
    """GT(Ground Truth) 데이터를 시각화하여 저장 (안전한 버전)
    
    Args:
        data_loader: 데이터 로더 (train 또는 val)
        class_names: 클래스 이름 리스트
        save_dir: 저장 디렉토리
        data_type: 데이터 타입 ("train" 또는 "val")
        max_batches: 시각화할 최대 배치 수
        max_images_per_batch: 배치당 최대 이미지 수
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting {data_type} GT data visualization (safe mode)...")
    
    batch_count = 0
    for batch_idx, batch in enumerate(data_loader):
        if batch_count >= max_batches:
            break
            
        print(f"Processing {data_type} batch {batch_idx + 1}/{min(max_batches, len(data_loader))}...")
        
        try:
            # 배치에서 이미지와 레이블 추출
            if isinstance(batch, dict):
                images = batch['images']
                labels = batch['labels']
                img_paths = batch.get('img_paths', None)
            else:
                images, labels = batch
                img_paths = None
            
            # 안전한 이미지 처리 (transform된 이미지 사용)
            batch_images = images[:max_images_per_batch]
            batch_labels = labels[:max_images_per_batch]
            empty_predictions = [None] * len(batch_images)
            
            # 시각화 및 저장
            save_path = save_dir / f'{data_type}_gt_batch_{batch_idx + 1}.png'
            
            # 안전한 시각화 함수 호출
            visualize_batch_safe(
                images=batch_images,
                targets=batch_labels,
                predictions=empty_predictions,
                class_names=class_names,
                save_path=save_path,
                max_images=max_images_per_batch
            )
            
            # 레이블 통계 출력
            total_objects = 0
            valid_images = 0
            for labels_per_img in batch_labels:
                if labels_per_img is not None and len(labels_per_img) > 0:
                    valid_images += 1
                    total_objects += len(labels_per_img)
            
            print(f"  - Batch {batch_idx + 1}: {len(batch_images)} images, {valid_images} with labels, {total_objects} total objects")
            
            batch_count += 1
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
    
    print(f"Completed {data_type} GT data visualization. Saved {batch_count} batches to {save_dir}")

def denormalize(img, mean, std):
    """정규화된 이미지를 원본으로 복원"""
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

def visualize_batch_safe(
    images: torch.Tensor,
    targets: List[torch.Tensor],
    predictions: List[torch.Tensor],
    class_names: List[str],
    save_path: Path,
    max_images: int = 8
) -> None:
    """안전한 배치 시각화 함수 (hang 방지)
    
    Args:
        images: 배치 이미지 텐서 (B, C, H, W)
        targets: 정답 박스 리스트
        predictions: 예측 박스 리스트 (사용하지 않음)
        class_names: 클래스 이름 리스트
        save_path: 저장 경로
        max_images: 시각화할 최대 이미지 수
    """
    try:
        # 배치 크기 제한
        batch_size = min(len(images), max_images)
        
        # 서브플롯 크기 계산
        n_cols = min(4, batch_size)
        n_rows = (batch_size - 1) // n_cols + 1
        
        plt.figure(figsize=(n_cols * 4, n_rows * 4))
        
        # 정규화 해제용 mean/std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        for i in range(batch_size):
            plt.subplot(n_rows, n_cols, i + 1)
            
            # 이미지 처리
            img = images[i]
            if torch.is_tensor(img):
                # 정규화 해제 후 시각화
                img = denormalize(img, mean, std)
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = np.clip(img_np, 0, 1)
            else:
                img_np = img
            
            plt.imshow(img_np)
            plt.axis('off')
            
            # 이미지 크기
            img_h, img_w = img_np.shape[:2]
            
            # 정답 박스 그리기 (녹색)
            if i < len(targets) and targets[i] is not None and len(targets[i]) > 0:
                target_boxes = targets[i].cpu().numpy() if torch.is_tensor(targets[i]) else targets[i]
                
                for box in target_boxes:
                    if len(box) >= 5:  # class_id, x_center, y_center, width, height
                        try:
                            class_id = int(box[0])
                            
                            # 클래스 ID 유효성 검증
                            if 0 <= class_id < len(class_names):
                                # center + width/height 형식 -> 픽셀 좌표 변환
                                x_center, y_center, width, height = box[1:5]
                                
                                # 픽셀 좌표 계산 (center -> top-left corner)
                                x1 = (x_center - width / 2) * img_w
                                y1 = (y_center - height / 2) * img_h
                                x2 = (x_center + width / 2) * img_w
                                y2 = (y_center + height / 2) * img_h
                                
                                # 경계 체크
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(img_w-1, x2), min(img_h-1, y2)
                                
                                # 박스 그리기
                                rect = plt.Rectangle(
                                    (x1, y1), x2-x1, y2-y1,
                                    fill=False, color='lime', linewidth=2, alpha=0.8
                                )
                                plt.gca().add_patch(rect)
                                
                                # 클래스 이름 표시 (간단하게)
                                plt.text(
                                    x1, y1-5,
                                    f'{class_names[class_id]}',
                                    color='lime',
                                    fontsize=8,
                                    fontweight='bold',
                                    bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7)
                                )
                        except Exception as e:
                            print(f"Warning: Error drawing box {box}: {e}")
                            continue
            
            plt.title(f'Image {i+1}', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error in visualize_batch_safe: {e}")
        # 빈 이미지 생성
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f'Visualization Error: {e}', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.savefig(save_path)
        plt.close()

def visualize_dataset_statistics(
    labeled_loader: torch.utils.data.DataLoader,
    unlabeled_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    class_names: List[str],
    save_dir: Path
) -> None:
    """데이터셋 통계 시각화 (간단한 버전)
    
    Args:
        labeled_loader: 레이블된 데이터 로더
        unlabeled_loader: 레이블되지 않은 데이터 로더
        val_loader: 검증 데이터 로더
        class_names: 클래스 이름 리스트
        save_dir: 저장 디렉토리
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 데이터셋 통계 시각화 중...")
    
    # 데이터셋 크기 통계
    dataset_sizes = {
        'Labeled': len(labeled_loader.dataset),
        'Unlabeled': len(unlabeled_loader.dataset),
        'Validation': len(val_loader.dataset)
    }
    
    # 클래스별 객체 수 통계 (레이블된 데이터에서만)
    class_counts = {name: 0 for name in class_names}
    total_objects = 0
    
    try:
        # 레이블된 데이터에서 클래스별 객체 수 계산
        for batch_idx, batch in enumerate(labeled_loader):
            if batch_idx >= 10:  # 샘플링으로 제한
                break
                
            labels = batch['labels']
            for labels_per_img in labels:
                if labels_per_img is not None and len(labels_per_img) > 0:
                    for label in labels_per_img:
                        if len(label) >= 1:
                            class_id = int(label[0])
                            if 0 <= class_id < len(class_names):
                                class_counts[class_names[class_id]] += 1
                                total_objects += 1
        
        # 통계 시각화
        plt.figure(figsize=(15, 10))
        
        # 1. 데이터셋 크기
        plt.subplot(2, 2, 1)
        sizes = list(dataset_sizes.values())
        labels = list(dataset_sizes.keys())
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Dataset Size Distribution', fontsize=14, fontweight='bold')
        
        # 2. 클래스별 객체 수 (상위 20개만)
        plt.subplot(2, 2, 2)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        class_names_plot = [item[0] for item in sorted_classes]
        class_counts_plot = [item[1] for item in sorted_classes]
        
        plt.barh(range(len(class_names_plot)), class_counts_plot, color='lightblue')
        plt.yticks(range(len(class_names_plot)), class_names_plot)
        plt.xlabel('Object Count')
        plt.title('Top 20 Classes by Object Count', fontsize=12)
        plt.gca().invert_yaxis()
        
        # 3. 객체 수 분포 히스토그램
        plt.subplot(2, 2, 3)
        non_zero_counts = [count for count in class_counts.values() if count > 0]
        plt.hist(non_zero_counts, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        plt.xlabel('Object Count per Class')
        plt.ylabel('Number of Classes')
        plt.title('Distribution of Object Counts per Class', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 4. 요약 통계
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        stats_text = f"""Dataset Statistics Summary:

Total Images:
• Labeled: {dataset_sizes['Labeled']:,}
• Unlabeled: {dataset_sizes['Unlabeled']:,}
• Validation: {dataset_sizes['Validation']:,}

Object Detection:
• Total Objects: {total_objects:,}
• Classes with Objects: {len([c for c in class_counts.values() if c > 0])}
• Average Objects per Class: {total_objects / len([c for c in class_counts.values() if c > 0]):.1f}

Top 5 Classes:
"""
        
        top_5 = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (class_name, count) in enumerate(top_5, 1):
            stats_text += f"• {class_name}: {count:,}\n"
        
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'dataset_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 데이터셋 통계 시각화 완료: {save_dir / 'dataset_statistics.png'}")
        
    except Exception as e:
        print(f"❌ 데이터셋 통계 시각화 실패: {e}")
        # 간단한 텍스트 파일로 통계 저장
        with open(save_dir / 'dataset_statistics.txt', 'w') as f:
            f.write("Dataset Statistics:\n")
            f.write("=" * 50 + "\n")
            for name, size in dataset_sizes.items():
                f.write(f"{name}: {size:,} images\n")
            f.write(f"\nTotal Objects: {total_objects:,}\n")
            f.write(f"Classes with Objects: {len([c for c in class_counts.values() if c > 0])}\n")
 
def plot_metrics(metrics: Dict[str, List[float]], save_path: Path) -> None:
    """학습 메트릭 시각화

    Args:
        metrics: 메트릭 딕셔너리
        save_path: 저장 경로
    """
    plt.figure(figsize=(12, 8))
    
    # Loss 그래프
    plt.subplot(2, 1, 1)
    plt.plot(metrics['epoch'], metrics['loss'], label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # mAP 그래프 (최종 평가 결과)
    plt.subplot(2, 1, 2)
    if 'mAP50' in metrics and 'mAP50-95' in metrics:
        plt.bar(['mAP50', 'mAP50-95'], [metrics['mAP50'][-1], metrics['mAP50-95'][-1]])
        plt.title('Final Evaluation Results')
        plt.ylabel('mAP')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_curves(
    metrics: Dict[str, List[float]], 
    save_path: Path,
    title: str = "Training Curves"
) -> None:
    """
    학습 곡선을 시각화하고 저장
    
    Args:
        metrics: 메트릭 딕셔너리 {'epoch': [...], 'loss': [...], 'mAP50': [...], ...}
        save_path: 저장 경로
        title: 차트 제목
    """
    plt.figure(figsize=(15, 10))
    
    # Loss 곡선
    if 'loss' in metrics and len(metrics['loss']) > 0:
        plt.subplot(2, 3, 1)
        plt.plot(metrics.get('epoch', range(len(metrics['loss']))), metrics['loss'], 'b-', linewidth=2)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
    
    # mAP 곡선들
    if 'mAP50' in metrics and len(metrics['mAP50']) > 0:
        plt.subplot(2, 3, 2)
        plt.plot(metrics.get('epoch', range(len(metrics['mAP50']))), metrics['mAP50'], 'g-', linewidth=2, label='mAP@0.5')
        if 'mAP50-95' in metrics and len(metrics['mAP50-95']) > 0:
            plt.plot(metrics.get('epoch', range(len(metrics['mAP50-95']))), metrics['mAP50-95'], 'r-', linewidth=2, label='mAP@0.5:0.95')
        plt.title('Detection Performance')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # MC Loss 컴포넌트들
    mc_components = ['epistemic_loss', 'variance_loss', 'entropy_loss', 'student_mc_loss']
    subplot_idx = 3
    
    for component in mc_components:
        if component in metrics and len(metrics[component]) > 0:
            plt.subplot(2, 3, subplot_idx)
            plt.plot(metrics.get('epoch', range(len(metrics[component]))), metrics[component], linewidth=2)
            plt.title(f'{component.replace("_", " ").title()}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss Value')
            plt.grid(True, alpha=0.3)
            subplot_idx += 1
            if subplot_idx > 6:
                break
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 저장
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved: {save_path}")

def plot_uncertainty_distribution(
    uncertainty_data: Dict[str, List[float]],
    save_path: Path,
    title: str = "Uncertainty Distribution"
) -> None:
    """
    불확실성 분포를 시각화하고 저장
    
    Args:
        uncertainty_data: 불확실성 데이터 {'box_variance': [...], 'class_entropy': [...], ...}
        save_path: 저장 경로
        title: 차트 제목
    """
    plt.figure(figsize=(15, 8))
    
    # Box Variance 분포
    if 'box_variance' in uncertainty_data and len(uncertainty_data['box_variance']) > 0:
        plt.subplot(2, 3, 1)
        plt.hist(uncertainty_data['box_variance'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Box Variance Distribution')
        plt.xlabel('Box Variance')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 통계 정보 추가
        variance_mean = np.mean(uncertainty_data['box_variance'])
        variance_std = np.std(uncertainty_data['box_variance'])
        plt.axvline(variance_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {variance_mean:.4f}')
        plt.axvline(variance_mean + variance_std, color='orange', linestyle=':', linewidth=2, label=f'+1σ: {variance_mean + variance_std:.4f}')
        plt.legend()
    
    # Class Entropy 분포
    if 'class_entropy' in uncertainty_data and len(uncertainty_data['class_entropy']) > 0:
        plt.subplot(2, 3, 2)
        plt.hist(uncertainty_data['class_entropy'], bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.title('Class Entropy Distribution')
        plt.xlabel('Class Entropy')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 통계 정보 추가
        entropy_mean = np.mean(uncertainty_data['class_entropy'])
        entropy_std = np.std(uncertainty_data['class_entropy'])
        plt.axvline(entropy_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {entropy_mean:.4f}')
        plt.axvline(entropy_mean + entropy_std, color='orange', linestyle=':', linewidth=2, label=f'+1σ: {entropy_mean + entropy_std:.4f}')
        plt.legend()
    
    # Confidence 분포
    if 'confidence' in uncertainty_data and len(uncertainty_data['confidence']) > 0:
        plt.subplot(2, 3, 3)
        plt.hist(uncertainty_data['confidence'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 신뢰도 임계값 표시
        confidence_threshold = 0.5
        plt.axvline(confidence_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {confidence_threshold}')
        plt.legend()
    
    # MC 샘플링 횟수별 분포 (있는 경우)
    if 'mc_samples' in uncertainty_data and len(uncertainty_data['mc_samples']) > 0:
        plt.subplot(2, 3, 4)
        unique_samples, counts = np.unique(uncertainty_data['mc_samples'], return_counts=True)
        plt.bar(unique_samples, counts, alpha=0.7, color='orange', edgecolor='black')
        plt.title('MC Sampling Count Distribution')
        plt.xlabel('Number of MC Samples')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    # Uncertainty Quality Score (조합 지표)
    if all(key in uncertainty_data for key in ['box_variance', 'class_entropy']):
        plt.subplot(2, 3, 5)
        # Quality Score = 1 / (1 + box_variance + class_entropy) (높을수록 좋음)
        quality_scores = [1 / (1 + bv + ce) for bv, ce in zip(uncertainty_data['box_variance'], uncertainty_data['class_entropy'])]
        plt.hist(quality_scores, bins=50, alpha=0.7, color='red', edgecolor='black')
        plt.title('Uncertainty Quality Score')
        plt.xlabel('Quality Score (higher = better)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        quality_mean = np.mean(quality_scores)
        plt.axvline(quality_mean, color='blue', linestyle='--', linewidth=2, label=f'Mean: {quality_mean:.4f}')
        plt.legend()
    
    # Box Variance vs Class Entropy 산점도
    if all(key in uncertainty_data for key in ['box_variance', 'class_entropy']):
        plt.subplot(2, 3, 6)
        plt.scatter(uncertainty_data['box_variance'], uncertainty_data['class_entropy'], 
                   alpha=0.6, s=10, c='darkgreen')
        plt.title('Box Variance vs Class Entropy')
        plt.xlabel('Box Variance')
        plt.ylabel('Class Entropy')
        plt.grid(True, alpha=0.3)
        
        # 상관관계 추가
        correlation = np.corrcoef(uncertainty_data['box_variance'], uncertainty_data['class_entropy'])[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 저장
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Uncertainty distribution saved: {save_path}")

def create_mc_loss_visualization(
    mc_loss_history: List[Dict],
    save_path: Path,
    title: str = "MC Loss Analysis"
) -> None:
    """
    MC Loss 컴포넌트들의 시간에 따른 변화를 시각화
    
    Args:
        mc_loss_history: MC Loss 히스토리 [{'epoch': 0, 'epistemic_loss': 0.1, ...}, ...]
        save_path: 저장 경로
        title: 차트 제목
    """
    if not mc_loss_history:
        print("Warning: Empty MC loss history provided")
        return
    
    plt.figure(figsize=(15, 10))
    
    # 데이터 추출
    epochs = [item['epoch'] for item in mc_loss_history]
    epistemic_losses = [item.get('epistemic_loss', 0) for item in mc_loss_history]
    variance_losses = [item.get('variance_loss', 0) for item in mc_loss_history]
    entropy_losses = [item.get('entropy_loss', 0) for item in mc_loss_history]
    total_losses = [item.get('total_loss', 0) for item in mc_loss_history]
    
    # 개별 컴포넌트 플롯
    plt.subplot(2, 2, 1)
    plt.plot(epochs, epistemic_losses, 'b-', linewidth=2, label='Epistemic Loss')
    plt.title('Epistemic Uncertainty Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, variance_losses, 'r-', linewidth=2, label='Variance Loss')
    plt.title('Predictive Variance Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, entropy_losses, 'g-', linewidth=2, label='Entropy Loss')
    plt.title('Entropy Regularization Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 전체 컴포넌트 비교
    plt.subplot(2, 2, 4)
    plt.plot(epochs, total_losses, 'k-', linewidth=3, label='Total MC Loss')
    plt.plot(epochs, epistemic_losses, 'b--', alpha=0.7, label='Epistemic')
    plt.plot(epochs, variance_losses, 'r--', alpha=0.7, label='Variance')
    plt.plot(epochs, entropy_losses, 'g--', alpha=0.7, label='Entropy')
    plt.title('All MC Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 저장
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ MC Loss visualization saved: {save_path}") 