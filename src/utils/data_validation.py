"""
데이터 검증 및 필터링 유틸리티
"""
import os
import glob
from pathlib import Path
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

def validate_and_filter_labels(labels_dir: str, num_classes: int = 80, backup_original: bool = True):
    """
    라벨 파일에서 잘못된 클래스 인덱스를 필터링
    
    Args:
        labels_dir: 라벨 파일이 있는 디렉토리
        num_classes: 유효한 클래스 수 (기본값: 80)
        backup_original: 원본 파일 백업 여부
    """
    labels_path = Path(labels_dir)
    if not labels_path.exists():
        logger.warning(f"Labels directory does not exist: {labels_dir}")
        return
    
    # 모든 .txt 라벨 파일 찾기
    label_files = list(labels_path.glob("*.txt"))
    logger.info(f"Found {len(label_files)} label files in {labels_dir}")
    
    filtered_count = 0
    invalid_count = 0
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            valid_lines = []
            file_modified = False
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # 최소 5개 값 (class, x_center, y_center, width, height)
                    try:
                        class_id = int(parts[0])
                        if 0 <= class_id < num_classes:
                            valid_lines.append(line)
                        else:
                            invalid_count += 1
                            file_modified = True
                    except ValueError:
                        invalid_count += 1
                        file_modified = True
                else:
                    invalid_count += 1
                    file_modified = True
            
            if file_modified:
                if backup_original:
                    backup_file = label_file.with_suffix('.txt.backup')
                    if not backup_file.exists():
                        label_file.rename(backup_file)
                
                with open(label_file, 'w') as f:
                    f.writelines(valid_lines)
                
                filtered_count += 1
                
        except Exception as e:
            logger.error(f"Error processing {label_file}: {e}")
    
    logger.info(f"Processed {len(label_files)} files, modified {filtered_count} files, filtered {invalid_count} invalid entries")

def validate_coco_labels(coco_root: str, num_classes: int = 80):
    """
    COCO 데이터셋의 라벨 파일들을 검증하고 필터링
    
    Args:
        coco_root: COCO 데이터셋 루트 디렉토리
        num_classes: 유효한 클래스 수
    """
    coco_path = Path(coco_root)
    
    # Train2017 라벨 검증
    train_labels = coco_path / "train2017" / "labels"
    if train_labels.exists():
        logger.info("Validating COCO train2017 labels...")
        validate_and_filter_labels(str(train_labels), num_classes)
    
    # Val2017 라벨 검증
    val_labels = coco_path / "val2017" / "labels"
    if val_labels.exists():
        logger.info("Validating COCO val2017 labels...")
        validate_and_filter_labels(str(val_labels), num_classes)

def validate_box_coordinates(boxes, image_shape=None):
    """
    박스 좌표를 검증하고 수정
    
    Args:
        boxes: 박스 좌표 (x1, y1, x2, y2) 또는 (x_center, y_center, width, height) 형식
        image_shape: 이미지 크기 (height, width), None이면 좌표만 검증
    
    Returns:
        validated_boxes: 검증된 박스 좌표
        valid_mask: 유효한 박스 마스크
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    
    boxes = np.array(boxes)
    if len(boxes.shape) == 1:
        boxes = boxes.reshape(1, -1)
    
    valid_mask = np.ones(len(boxes), dtype=bool)
    validated_boxes = boxes.copy()
    
    for i, box in enumerate(boxes):
        if len(box) >= 4:
            if len(box) == 4:  # (x1, y1, x2, y2) 형식
                x1, y1, x2, y2 = box[:4]
                
                # 좌표 순서 검증 및 수정
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                # 음수 좌표 처리
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = max(x1 + 1, x2)  # 최소 1픽셀 너비 보장
                y2 = max(y1 + 1, y2)  # 최소 1픽셀 높이 보장
                
                validated_boxes[i, :4] = [x1, y1, x2, y2]
                
                # 이미지 크기 제한
                if image_shape is not None:
                    height, width = image_shape[:2]
                    validated_boxes[i, 0] = min(width - 1, validated_boxes[i, 0])  # x1
                    validated_boxes[i, 1] = min(height - 1, validated_boxes[i, 1])  # y1
                    validated_boxes[i, 2] = min(width, validated_boxes[i, 2])       # x2
                    validated_boxes[i, 3] = min(height, validated_boxes[i, 3])      # y2
                
            elif len(box) >= 5:  # (x_center, y_center, width, height, ...) 형식
                x_center, y_center, width, height = box[:4]
                
                # 음수 크기 처리
                width = max(1, abs(width))
                height = max(1, abs(height))
                
                # 중심점이 이미지 밖에 있는 경우 처리
                if image_shape is not None:
                    img_height, img_width = image_shape[:2]
                    x_center = np.clip(x_center, 0, img_width)
                    y_center = np.clip(y_center, 0, img_height)
                    
                    # 박스가 이미지 경계를 벗어나지 않도록 조정
                    half_width = width / 2
                    half_height = height / 2
                    
                    x1 = max(0, x_center - half_width)
                    y1 = max(0, y_center - half_height)
                    x2 = min(img_width, x_center + half_width)
                    y2 = min(img_height, y_center + half_height)
                    
                    # 새로운 중심점과 크기 계산
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                
                validated_boxes[i, :4] = [x_center, y_center, width, height]
        
        # 너무 작은 박스 제거
        if len(box) >= 4:
            if len(box) == 4:  # (x1, y1, x2, y2)
                width = validated_boxes[i, 2] - validated_boxes[i, 0]
                height = validated_boxes[i, 3] - validated_boxes[i, 1]
            else:  # (x_center, y_center, width, height)
                width = validated_boxes[i, 2]
                height = validated_boxes[i, 3]
            
            # 최소 크기 검증 (3x3 픽셀)
            if width < 3 or height < 3:
                valid_mask[i] = False
    
    return validated_boxes, valid_mask

def filter_invalid_predictions(predictions, image_shape=None, min_size=3):
    """
    MC Dropout 예측 결과에서 잘못된 박스들을 필터링
    
    Args:
        predictions: YOLO 예측 결과 (boxes, scores, class_ids)
        image_shape: 이미지 크기
        min_size: 최소 박스 크기
    
    Returns:
        filtered_predictions: 필터링된 예측 결과
    """
    if predictions is None or len(predictions) == 0:
        return predictions
    
    # 박스 좌표 검증
    if hasattr(predictions, 'boxes') and predictions.boxes is not None:
        boxes = predictions.boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2) 형식
        validated_boxes, valid_mask = validate_box_coordinates(boxes, image_shape)
        
        # 유효한 박스만 선택
        if np.any(valid_mask):
            filtered_predictions = predictions[valid_mask]
            return filtered_predictions
        else:
            # 모든 박스가 유효하지 않은 경우 빈 결과 반환
            return None
    
    return predictions

def validate_yolo_predictions(yolo_results, image_shape=None):
    """
    YOLO 예측 결과의 박스 좌표를 검증하고 수정
    
    Args:
        yolo_results: YOLO 모델의 예측 결과
        image_shape: 이미지 크기 (height, width)
    
    Returns:
        validated_results: 검증된 예측 결과
    """
    if yolo_results is None:
        return yolo_results
    
    try:
        # YOLO 결과에서 박스 정보 추출
        if hasattr(yolo_results, 'boxes') and yolo_results.boxes is not None:
            boxes = yolo_results.boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2) 형식
            
            if len(boxes) > 0:
                # 박스 좌표 검증
                validated_boxes, valid_mask = validate_box_coordinates(boxes, image_shape)
                
                if np.any(valid_mask):
                    # 유효한 박스만 유지
                    valid_indices = np.where(valid_mask)[0]
                    
                    # YOLO 결과 객체 수정
                    yolo_results.boxes.xyxy = torch.from_numpy(validated_boxes[valid_mask]).to(yolo_results.boxes.xyxy.device)
                    yolo_results.boxes.conf = yolo_results.boxes.conf[valid_indices]
                    yolo_results.boxes.cls = yolo_results.boxes.cls[valid_indices]
                    
                    if hasattr(yolo_results.boxes, 'id') and yolo_results.boxes.id is not None:
                        yolo_results.boxes.id = yolo_results.boxes.id[valid_indices]
                else:
                    # 모든 박스가 유효하지 않은 경우 빈 결과로 설정
                    yolo_results.boxes.xyxy = torch.empty(0, 4, device=yolo_results.boxes.xyxy.device)
                    yolo_results.boxes.conf = torch.empty(0, device=yolo_results.boxes.conf.device)
                    yolo_results.boxes.cls = torch.empty(0, device=yolo_results.boxes.cls.device)
                    if hasattr(yolo_results.boxes, 'id') and yolo_results.boxes.id is not None:
                        yolo_results.boxes.id = torch.empty(0, device=yolo_results.boxes.id.device)
        
        # 리스트 형태의 결과 처리
        elif isinstance(yolo_results, list):
            validated_list = []
            for result in yolo_results:
                validated_result = validate_yolo_predictions(result, image_shape)
                if validated_result is not None:
                    validated_list.append(validated_result)
            return validated_list
        
    except Exception as e:
        logger.warning(f"YOLO 예측 결과 검증 중 오류 발생: {e}")
        # 검증 실패 시 원본 반환
        pass
    
    return yolo_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate COCO labels")
    parser.add_argument("--coco_root", type=str, required=True, help="COCO dataset root directory")
    parser.add_argument("--num_classes", type=int, default=80, help="Number of classes")
    
    args = parser.parse_args()
    
    validate_coco_labels(args.coco_root, args.num_classes) 