import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from typing import List, Tuple, Dict

def visualize_batch(
    images: torch.Tensor,
    targets: List[torch.Tensor],
    predictions: List[torch.Tensor],
    class_names: List[str],
    save_path: Path,
    max_images: int = 16
) -> None:
    """배치 이미지와 예측 결과를 시각화하여 저장
    
    Args:
        images: 배치 이미지 텐서 (B, C, H, W)
        targets: 정답 박스 리스트 [(N, 5), ...], 5 = (class_id, x, y, w, h)
        predictions: 예측 박스 리스트 [(M, 6), ...], 6 = (x, y, w, h, conf, class_id)
        class_names: 클래스 이름 리스트
        save_path: 저장 경로
        max_images: 시각화할 최대 이미지 수
    """
    # 배치 크기 제한
    batch_size = min(len(images), max_images)
    
    # 서브플롯 크기 계산
    n_cols = min(4, batch_size)
    n_rows = (batch_size - 1) // n_cols + 1
    
    plt.figure(figsize=(n_cols * 5, n_rows * 5))
    
    for i in range(batch_size):
        # 이미지 변환 (텐서 -> numpy)
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 서브플롯 생성
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # 정답 박스 그리기 (녹색)
        if targets[i] is not None:
            for box in targets[i]:
                class_id = int(box[0])
                x, y, w, h = box[1:5]
                x1 = (x - w/2) * img.shape[1]
                y1 = (y - h/2) * img.shape[0]
                x2 = (x + w/2) * img.shape[1]
                y2 = (y + h/2) * img.shape[0]
                plt.gca().add_patch(plt.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    fill=False, color='g', linewidth=2
                ))
                plt.text(
                    x1, y1-5,
                    class_names[class_id],
                    color='g',
                    fontsize=8
                )
        
        # 예측 박스 그리기 (빨간색)
        if predictions[i] is not None:
            for box in predictions[i]:
                x1, y1, x2, y2, conf, class_id = box
                class_id = int(class_id)
                plt.gca().add_patch(plt.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    fill=False, color='r', linewidth=2
                ))
                plt.text(
                    x1, y1-15,
                    f'{class_names[class_id]} {conf:.2f}',
                    color='r',
                    fontsize=8
                )
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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