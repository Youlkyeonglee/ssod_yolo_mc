from pathlib import Path
import argparse
import torch
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경에서 사용
from models.yolo_mc import YOLOWithMCDropout
from uncertainty.mc_dropout import MCDropoutDetector, MCLoss
from data.semi_supervised_dataset import SemiSupervisedDataset
from utils.train_utils import (
    setup_logger,
    save_checkpoint,
    load_checkpoint,
    update_pseudo_labels,
    evaluate_model,
    calculate_class_wise_performance,
    save_class_performance_csv
)
from utils.yolo_losses import create_yolo_loss
import torchvision.transforms as transforms
import yaml
from tqdm import tqdm
from utils.visualization import visualize_batch, plot_metrics, visualize_gt_data, plot_training_curves, plot_uncertainty_distribution

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO with MC Dropout for Semi-Supervised Learning')
    
    # 기본 설정
    parser.add_argument('--config', type=str, 
                      default=str(Path(__file__).parent / 'configs' / 'yolo_config.yaml'),
                      help='YAML 설정 파일 경로')
    
    # 실행 관련 설정만 명령행 인자로 받음
    parser.add_argument('--save_dir', type=str, default=".",
                      help='프로젝트 루트 디렉토리 (runs 폴더가 생성될 위치)')
    parser.add_argument('--val_data_path', type=str, default="/home/lee/research/Research2025/ssod_yolo_mc/src/configs/coco_val.yaml",
                      help='검증 데이터 경로')
    parser.add_argument('--device', type=str, default=None,
                      help='실행 디바이스 (예: cpu, cuda:0)')
    parser.add_argument('--resume', type=str, default=None,
                      help='체크포인트에서 재시작')
    
    args = parser.parse_args()
    
    # YAML 설정 파일 로드
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # 설정값 정수형 변환 및 변수 참조 처리
    if isinstance(config['training']['semi_supervised']['pseudo_label_start_epoch'], str):
        if config['training']['semi_supervised']['pseudo_label_start_epoch'] == '${training.mature_epoch}':
            config['training']['semi_supervised']['pseudo_label_start_epoch'] = config['training']['mature_epoch']
    
    # config의 값들을 args에 추가
    args.model = config['model']['name']
    args.dropout_rate = config['model']['dropout']['rate']
    args.num_samples = config['model']['dropout']['num_samples']
    args.batch_size = config['data']['batch_size']
    args.epochs = config['training']['epochs']
    args.val_interval = config['training']['val_interval']
    args.save_interval = config['training']['save_interval']
    args.labeled_ratio = config['data']['labeled_ratio']
    args.seed = config['data']['seed']
    args.max_samples = config['data']['max_samples']
    args.box_std_threshold = config['training']['semi_supervised']['uncertainty']['box_std_threshold']
    args.entropy_threshold = config['training']['semi_supervised']['uncertainty']['entropy_threshold']
    args.uncertainty_weight = config['training']['uncertainty_weight']
    args.pseudo_label_weight = config['training']['semi_supervised']['pseudo_label_weight']
    args.pseudo_label_start_epoch = config['training']['semi_supervised']['pseudo_label_start_epoch']
    args.conf_threshold = config['training']['semi_supervised']['conf_threshold']
    args.feature_alignment_enabled = config['model']['feature_alignment']['enabled']
    args.feature_alignment_weight = config['model']['feature_alignment']['weight']
    args.num_workers = config['data']['num_workers']
    
    
    # device가 설정되지 않은 경우 config에서 가져오기
    if args.device is None:
        args.device = config['inference']['device']
    
    return args, config

def get_weak_augmentation(img_size: int = 640):
    """Weak Augmentation for Teacher model (안정적인 pseudo label 생성)"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_strong_augmentation(img_size: int = 640):
    """Strong Augmentation for Student model (robust 특징 학습)"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        # Cutout augmentation (randomly mask patches)
        transforms.Lambda(lambda x: apply_cutout(x, n_holes=1, length=32)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transform(img_size: int = 640):
    """Validation transform (no augmentation)"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def apply_cutout(img, n_holes=1, length=32):
    """
    Cutout augmentation: randomly mask square regions
    
    Args:
        img: PIL Image or tensor
        n_holes: number of holes to cut
        length: length of the square hole
    """
    if isinstance(img, torch.Tensor):
        c, h, w = img.shape
        mask = torch.ones((c, h, w), dtype=img.dtype, device=img.device)
        
        for _ in range(n_holes):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()
            
            y1 = max(0, y - length // 2)
            y2 = min(h, y + length // 2)
            x1 = max(0, x - length // 2)
            x2 = min(w, x + length // 2)
            
            mask[:, y1:y2, x1:x2] = 0
        
        img = img * mask
    
    return img

def get_transform(train: bool = True, img_size: int = 640, augmentation_type: str = "weak"):
    """
    데이터 변환 함수 생성 (Teacher-Student augmentation 전략)
    
    Args:
        train: 학습 모드 여부
        img_size: 이미지 크기
        augmentation_type: "weak", "strong", "none"
    """
    if not train:
        return get_val_transform(img_size)
    
    if augmentation_type == "weak":
        return get_weak_augmentation(img_size)
    elif augmentation_type == "strong":
        return get_strong_augmentation(img_size)
    else:
        return get_val_transform(img_size)

def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    정규화된 텐서를 원본 이미지로 복원
    
    Args:
        tensor: 정규화된 이미지 텐서 (C, H, W)
        mean: 정규화에 사용된 평균값
        std: 정규화에 사용된 표준편차
    
    Returns:
        denormalized tensor
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).view(-1, 1, 1)
    if isinstance(std, list):
        std = torch.tensor(std).view(-1, 1, 1)
    
    # GPU 텐서인 경우 CPU로 이동
    if tensor.is_cuda:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    
    # Denormalize: x = x * std + mean
    denormalized = tensor * std + mean
    
    # 0-1 범위로 클램핑
    denormalized = torch.clamp(denormalized, 0, 1)
    
    return denormalized

def tensor_to_pil(tensor):
    """
    정규화된 텐서를 PIL Image로 변환
    
    Args:
        tensor: 정규화된 이미지 텐서 (C, H, W)
    
    Returns:
        PIL Image
    """
    # Denormalize
    denormalized = denormalize_tensor(tensor)
    
    # PIL Image로 변환
    pil_image = transforms.ToPILImage()(denormalized.cpu())
    
    return pil_image

def train_one_epoch(
    model: torch.nn.Module,
    labeled_loader: torch.utils.data.DataLoader,
    unlabeled_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    detector: MCDropoutDetector,
    mc_loss_fn: MCLoss,  # MC Loss 함수 추가
    device: str,
    epoch: int,
    logger: logging.Logger,
    config: dict,
    pseudo_label_weight: float = 0.5,
    pseudo_label_start_epoch: int = 10,
    conf_threshold: float = 0.5,
    uncertainty_weight: float = 50.0,
    alignment_weight: float = 0.1  # Feature Alignment Loss 가중치
):
    """한 에포크 학습"""
    model.train()
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    labeled_iter = iter(labeled_loader)
    
    # MC Dropout 불확실성 임계값 설정 로드
    box_std_threshold = config['training']['semi_supervised']['uncertainty']['box_std_threshold']
    entropy_threshold = config['training']['semi_supervised']['uncertainty']['entropy_threshold']
    
    # MC Dropout 결과 저장을 위한 리스트 초기화
    mc_results_list = []
    
    # 의사 레이블 생성
    if epoch >= pseudo_label_start_epoch:
        pseudo_labels = update_pseudo_labels(
            model=model,
            unlabeled_loader=unlabeled_loader,
            detector=detector,
            conf_threshold=conf_threshold,
            device=device,
            config=config
        )
        logger.info(f"Generated {len(pseudo_labels)} pseudo labels")
    
    # 학습 루프
    num_batches = min(len(labeled_loader), len(unlabeled_loader))
    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}", dynamic_ncols=True, leave=False)
    for batch_idx in pbar:
        # 레이블된 데이터로부터 배치 가져오기
        try:
            labeled_batch = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            labeled_batch = next(labeled_iter)
        
        # Student 모델용 Strong Augmentation 적용
        strong_transform = get_strong_augmentation(config['data']['img_size'])
        
        # Labeled 데이터 처리 (이미 정규화된 상태로 사용)
        labeled_images = labeled_batch['images'].to(device)
        
        # 이미지 값이 0-1 범위에 있는지 확인하고 필요시 정규화
        if labeled_images.max() > 1.0:
            labeled_images = labeled_images / 255.0
        
        # YOLO 형식으로 targets 변환
        labeled_targets = []
        
        # 기존 로직
        for i, labels in enumerate(labeled_batch['labels']):
            if len(labels) > 0 and labels.shape[1] == 5:  # 유효한 레이블인지 확인
                batch_labels = torch.zeros((len(labels), 6), device=device)
                batch_labels[:, 0] = i  # batch index
                
                # 안전한 라벨 변환
                labels_safe = labels.to(device)
                
                # 클래스 ID 유효성 검증 (0 ~ num_classes-1)
                num_classes = config['data']['nc']
                class_ids = labels_safe[:, 0].long()
                class_ids = torch.clamp(class_ids, 0, num_classes - 1)
                
                # bbox 좌표 유효성 검증 (0.0 ~ 1.0)
                bbox_coords = labels_safe[:, 1:5]
                bbox_coords = torch.clamp(bbox_coords, 0.0, 1.0)
                
                # 안전한 값으로 할당
                batch_labels[:, 1] = class_ids.float()
                batch_labels[:, 2:6] = bbox_coords
                
                labeled_targets.append(batch_labels)
        
        if labeled_targets:
            labeled_targets = torch.cat(labeled_targets, dim=0)
            
            # YOLOv8 모델에서 순전파 수행
            model.student_model.model.train()
            
            # 예측 수행
            predictions = model.student_model.model(labeled_images)
            
            # YOLO 손실 함수를 사용하여 손실 계산
            if isinstance(predictions, (list, tuple)):
                # 다중 스케일 예측의 경우 첫 번째 예측만 사용
                pred = predictions[0] if len(predictions) > 0 else None
            else:
                pred = predictions
            
            # 간단한 loss 계산 (디버깅 메시지 제거)
            try:
                # YOLO 모델의 compute_loss 메서드 사용
                labeled_loss = model.student_model.model.compute_loss(predictions, labeled_targets)
                loss_dict = {'labeled_loss': labeled_loss.item()}
            except Exception as e:
                # Fallback to simple MSE loss
                try:
                    target_shape = pred.shape
                    dummy_target = torch.zeros(target_shape, device=device)
                    labeled_loss = torch.nn.functional.mse_loss(pred, dummy_target)
                    loss_dict = {'labeled_loss': labeled_loss.item()}
                except Exception:
                    labeled_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    loss_dict = {'labeled_loss': 0.0}
        else:
            labeled_loss = torch.tensor(0.0, device=device, requires_grad=True)
            loss_dict = {'labeled_loss': 0.0}
        
        # Feature Alignment Loss 추가
        if alignment_weight > 0.0:
            model_output = model(labeled_images)
            predictions = model_output['predictions']
            alignment_loss = model_output['alignment_loss']
            
            # alignment_loss가 None이거나 gradient가 없는 경우 처리
            if alignment_loss is not None and isinstance(alignment_loss, torch.Tensor):
                if alignment_loss.requires_grad:
                    labeled_loss = labeled_loss + alignment_weight * alignment_loss
                    loss_dict['alignment_loss'] = alignment_loss.item()
        
        # 🔬 Student MC Loss 추가 (새로운 불확실성 기반 손실 함수)
        student_mc_loss = 0.0
        consistency_weight = config['training']['semi_supervised']['consistency_weight']
        
        if consistency_weight > 0.0 and labeled_targets.shape[0] > 0:
            try:
                # Student 모델로 MC Dropout 다중 예측 수행
                mc_predictions = []
                model.student_model.model.train()  # Dropout 활성화 (PyTorch 모드 설정)
                
                for _ in range(config['training']['semi_supervised']['num_mc_samples']):
                    with torch.no_grad():
                        mc_pred = model.student_model(labeled_images)
                        if isinstance(mc_pred, (list, tuple)) and len(mc_pred) > 0:
                            mc_predictions.append(mc_pred[0])  # 첫 번째 scale만 사용
                        elif isinstance(mc_pred, torch.Tensor):
                            mc_predictions.append(mc_pred)
                
                if len(mc_predictions) >= 2:
                    # MC Loss 계산 (새로운 불확실성 기반 손실 함수)
                    mc_loss_results = mc_loss_fn(mc_predictions, reduction='mean')
                    student_mc_loss = mc_loss_results['mc_loss'] * consistency_weight
                    
                    # labeled_loss에 MC loss 추가
                    labeled_loss = labeled_loss + student_mc_loss
                    
                    # Loss 컴포넌트 추적
                    loss_dict['student_mc_loss'] = student_mc_loss.item()
                    loss_dict['epistemic_loss'] = mc_loss_results['epistemic_loss'].item()
                    loss_dict['variance_loss'] = mc_loss_results['variance_loss'].item()
                    loss_dict['entropy_loss'] = mc_loss_results['entropy_loss'].item()
                    
                    # 적응적 가중치 로깅
                    adaptive_weights = mc_loss_results['adaptive_weights']
                    if epoch % 10 == 0:  # 10 에포크마다 로깅
                        logger.debug(f"MC Loss adaptive weights - α: {adaptive_weights['alpha']:.3f}, "
                                   f"β: {adaptive_weights['beta']:.3f}, γ: {adaptive_weights['gamma']:.3f}")
                        
            except Exception as e:
                logger.debug(f"Student MC Loss calculation failed: {e}")
                student_mc_loss = 0.0
        
        # 레이블되지 않은 데이터 학습 (Teacher-Student MC Dropout 전략)
        if epoch >= pseudo_label_start_epoch and pseudo_labels:
            try:
                unlabeled_batch = next(iter(unlabeled_loader))
                
                # === Teacher MC Dropout: Weak Augmentation으로 안정적인 pseudo label 생성 ===
                weak_transform = get_weak_augmentation(config['data']['img_size'])
                weak_unlabeled_images = []
                for img in unlabeled_batch['images']:
                    if isinstance(img, torch.Tensor):
                        img_pil = tensor_to_pil(img)
                    else:
                        img_pil = img
                    weak_img = weak_transform(img_pil)
                    weak_unlabeled_images.append(weak_img)
                
                weak_unlabeled_images = torch.stack(weak_unlabeled_images).to(device)
                
                # Teacher MC Dropout 다중 예측
                try:
                    # === 기존 중복 MC 샘플링 제거 ===
                    # for mc_idx in range(num_mc_samples):
                    #     mc_result = detector.predict_with_uncertainty(weak_images, device=device)
                    #     if mc_result:
                    #         mc_results_list.append(mc_result)
                    
                    # === 개선된 단일 MC Dropout 호출 ===
                    # predict_with_uncertainty 함수가 내부적으로 이미 num_samples만큼 MC 샘플링을 수행함
                    mc_result = detector.predict_with_uncertainty(weak_unlabeled_images, device=device)
                    
                    if mc_result:
                        # 기존 코드와의 호환성을 위해 리스트로 래핑
                        mc_results_list = [mc_result]
                        
                        print(f"  📊 MC Dropout completed: {len(mc_result)} image results with {detector.num_samples} internal samples each")
                    
                    if len(mc_results_list) >= 1:  # 최소 1개 결과 필요
                        # === MC Dropout 불확실성 기반 고품질 Pseudo Label 선별 ===
                        
                        # 각 detection에 대해 MC 샘플들의 분산 계산
                        high_quality_pseudo_labels = []
                        
                        for img_idx in range(len(weak_unlabeled_images)):
                            # 해당 이미지의 모든 MC 샘플 수집
                            img_mc_detections = []
                            
                            for mc_results in mc_results_list:
                                if img_idx < len(mc_results):
                                    result = mc_results[img_idx]
                                    if len(result.get('boxes', [])) > 0:
                                        img_mc_detections.append(result)
                            
                            # MC Dropout에서 한 번의 호출로도 내부적으로 여러 샘플을 비교하므로
                            # 최소 조건을 1개로 낮춤
                            if len(img_mc_detections) >= 1:  
                                # Detection들의 일관성 분석 (단일 결과라도 불확실성 지표 활용)
                                consistent_detections = extract_confident_detections(
                                    img_mc_detections[0],  # 첫 번째 (유일한) MC 결과 사용
                                    box_std_threshold=box_std_threshold,
                                    entropy_threshold=entropy_threshold,
                                    conf_threshold=conf_threshold
                                )
                                
                                if len(consistent_detections) > 0:
                                    high_quality_pseudo_labels.append({
                                        'boxes': consistent_detections,
                                        'image_path': unlabeled_batch.get('paths', [''])[img_idx] if 'paths' in unlabeled_batch and img_idx < len(unlabeled_batch.get('paths', [])) else '',
                                        'uncertainty_stats': {
                                            'mc_samples': detector.num_samples,  # 내부 MC 샘플 수
                                            'detections_count': len(consistent_detections)
                                        }
                                    })
                        
                        pseudo_labels.extend(high_quality_pseudo_labels)
                        
                        if high_quality_pseudo_labels:
                            total_detections = sum(len(pl['boxes']) for pl in high_quality_pseudo_labels)
                            print(f"  📊 Current batch: {len(high_quality_pseudo_labels)} images, {total_detections} high-quality detections")
                    
                except Exception as e:
                    print(f"❌ Error in Teacher MC Dropout pseudo label generation: {e}")
                    total_loss = labeled_loss
                    continue
                
                # pseudo label 처리가 성공적으로 완료된 경우 total_loss 설정
                total_loss = labeled_loss
            
            except Exception as e:
                logger.info(f"Error in Teacher-Student MC Dropout calculation: {e}")
                total_loss = labeled_loss
        else:
            total_loss = labeled_loss
        
        # total_loss 안전장치 - 초기 텐서 상태인 경우 labeled_loss로 설정
        if total_loss.item() == 0.0 and hasattr(total_loss, 'grad_fn') and total_loss.grad_fn is None:
            total_loss = labeled_loss
            logger.debug(f"Batch {batch_idx}: total_loss was unset, using labeled_loss: {labeled_loss.item():.6f}")
        
        # 역전파
        optimizer.zero_grad()
        
        # gradient 체크 및 안전장치
        if total_loss.requires_grad:
            total_loss.backward()
            optimizer.step()
            
            # 🔄 EMA Teacher 업데이트 (매 배치마다)
            if hasattr(model, 'update_teacher_ema'):
                try:
                    model.update_teacher_ema()
                except RuntimeError as e:
                    logger.warning(f"⚠️  Teacher EMA update failed: {e}")
                    logger.warning("   Continuing training without teacher update...")
                except Exception as e:
                    logger.warning(f"⚠️  Unexpected error in teacher update: {e}")
                    logger.warning("   Continuing training without teacher update...")
        else:
            # total_loss가 gradient를 요구하지 않는 경우 건너뜀
            pass
        
        # Loss 값 검증 및 클리핑
        if total_loss.item() > 100.0:
            logger.warning(f"🚨 High loss detected: {total_loss.item():.2f} - applying gradient clipping")
            # 극도로 높은 loss의 경우 gradient clipping 적용
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        # 현재 배치의 loss 값을 tqdm description에 업데이트
        desc = f"Epoch {epoch}"
        if loss_dict:
            desc += f" - Loss: {total_loss.item():.4f}"
            for k, v in loss_dict.items():
                desc += f", {k}: {v:.4f}"
        pbar.set_description(desc)
        
        # 🎯 Student MC Loss 로깅 (배치별)
        if 'student_mc_loss' in loss_dict:
            if batch_idx % 10 == 0:  # 10배치마다 로깅
                logger.debug(f"Batch {batch_idx}: Student MC Loss = {loss_dict['student_mc_loss']:.6f}")
    
    # 🔬 MC Dropout 품질 모니터링 (매 에포크마다)
    mc_quality_stats = evaluate_mc_dropout_quality(
        model=model,
        data_loader=labeled_loader,
        detector=detector,
        device=device,
        epoch=epoch,
        logger=logger,
        config=config
    )
    
    # MC Dropout 통계를 로그에 추가
    if mc_quality_stats:
        logger.info(f"🔬 [Epoch {epoch}] MC Dropout Quality Metrics:")
        for metric, value in mc_quality_stats.items():
            # 값의 타입에 따라 적절한 포맷 적용
            if isinstance(value, (int, float)):
                logger.info(f"  - {metric}: {value:.6f}")
            else:
                logger.info(f"  - {metric}: {value}")  # 문자열인 경우 그대로 출력
    
    return total_loss.item()

def evaluate_mc_dropout_quality(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    detector: MCDropoutDetector,
    device: str,
    epoch: int,
    logger: logging.Logger,
    config: dict,
    max_batches: int = 3  # 빠른 평가를 위해 3개 배치만 사용
):
    """매 에포크마다 MC Dropout 품질 평가
    
    Args:
        model: 평가할 모델
        data_loader: 데이터 로더
        detector: MC Dropout 탐지기
        device: 실행 디바이스
        epoch: 현재 에포크
        logger: 로거
        config: 설정
        max_batches: 평가할 최대 배치 수
    
    Returns:
        MC Dropout 품질 통계
    """
    model.eval()
    
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
                uncertainty_results = detector.predict_with_uncertainty(images)
                
                if uncertainty_results:
                    detection_counts.append(len(uncertainty_results))
                    
                    # 불확실성 지표 수집
                    for result in uncertainty_results:
                        if 'box_variance' in result:
                            box_var = result['box_variance']
                            if hasattr(box_var, 'item'):
                                if box_var.numel() > 0:
                                    box_variances.append(box_var.item())
                            elif isinstance(box_var, (int, float)):
                                box_variances.append(box_var)
                        
                        if 'class_entropy' in result:
                            class_ent = result['class_entropy']
                            if hasattr(class_ent, 'item'):
                                if class_ent.numel() > 0:
                                    class_entropies.append(class_ent.item())
                            elif isinstance(class_ent, (int, float)):
                                class_entropies.append(class_ent)
                        
                        if 'confidence' in result:
                            conf = result['confidence']
                            if hasattr(conf, 'item'):
                                confidence_scores.append(conf.item())
                            elif isinstance(conf, (int, float)):
                                confidence_scores.append(conf)
                else:
                    detection_counts.append(0)
                
                batch_count += 1
                
            except Exception as e:
                logger.warning(f"MC Dropout quality evaluation error in batch {batch_count}: {e}")
                detection_counts.append(0)
                batch_count += 1
    
    # 통계 계산
    quality_stats = {}
    
    if box_variances:
        quality_stats['avg_box_variance'] = sum(box_variances) / len(box_variances)
        quality_stats['min_box_variance'] = min(box_variances)
        quality_stats['max_box_variance'] = max(box_variances)
    else:
        quality_stats['avg_box_variance'] = 0.0
    
    if class_entropies:
        quality_stats['avg_class_entropy'] = sum(class_entropies) / len(class_entropies)
        quality_stats['min_class_entropy'] = min(class_entropies)
        quality_stats['max_class_entropy'] = max(class_entropies)
    else:
        quality_stats['avg_class_entropy'] = 0.0
    
    if confidence_scores:
        quality_stats['avg_confidence'] = sum(confidence_scores) / len(confidence_scores)
        quality_stats['min_confidence'] = min(confidence_scores)
        quality_stats['max_confidence'] = max(confidence_scores)
    else:
        quality_stats['avg_confidence'] = 0.0
    
    if detection_counts:
        quality_stats['avg_detections_per_image'] = sum(detection_counts) / len(detection_counts)
        quality_stats['total_detections'] = sum(detection_counts)
    else:
        quality_stats['avg_detections_per_image'] = 0.0
        quality_stats['total_detections'] = 0
    
    # MC Dropout 설정 정보 추가
    quality_stats['mc_num_samples'] = config['model']['dropout']['num_samples']
    quality_stats['mc_dropout_rate'] = config['model']['dropout']['rate']
    
    # 품질 향상 지표 계산 (에포크 10 이후부터)
    if epoch >= 10:
        # Box variance가 낮을수록 좋음
        if quality_stats['avg_box_variance'] < 0.1:
            quality_stats['box_quality_grade'] = 'Excellent'
        elif quality_stats['avg_box_variance'] < 0.2:
            quality_stats['box_quality_grade'] = 'Good'
        elif quality_stats['avg_box_variance'] < 0.3:
            quality_stats['box_quality_grade'] = 'Fair'
        else:
            quality_stats['box_quality_grade'] = 'Poor'
        
        # Confidence가 높을수록 좋음
        if quality_stats['avg_confidence'] > 0.8:
            quality_stats['confidence_quality_grade'] = 'Excellent'
        elif quality_stats['avg_confidence'] > 0.6:
            quality_stats['confidence_quality_grade'] = 'Good'
        elif quality_stats['avg_confidence'] > 0.4:
            quality_stats['confidence_quality_grade'] = 'Fair'
        else:
            quality_stats['confidence_quality_grade'] = 'Poor'
    
    return quality_stats

def get_run_dir(args, model_name: str, labeled_ratio: float) -> Path:
    """실행 디렉토리 생성 및 반환
    
    Args:
        model_name: YOLO 모델 이름 (e.g., "yolov8m")
        labeled_ratio: 레이블된 데이터 비율
    
    Returns:
        실행 디렉토리 경로
    """
    # runs/train 디렉토리 생성 (YOLO 표준 구조)
    save_dir = getattr(args, 'save_dir', '.')
    runs_dir = Path(save_dir) / "runs"
    base_dir = runs_dir / "train"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 기본 실행 디렉토리 이름 생성
    run_name = f"{model_name}_label_p{labeled_ratio}"
    
    # 이미 존재하는 디렉토리 확인
    existing_runs = list(base_dir.glob(f"{run_name}*"))
    if not existing_runs:
        run_dir = base_dir / run_name
    else:
        # 마지막 번호 찾기
        max_num = 0
        for run in existing_runs:
            if run.name == run_name:
                max_num = 1
            else:
                try:
                    num = int(run.name.split("_")[-1])
                    max_num = max(max_num, num)
                except ValueError:
                    continue
        
        # 새 디렉토리 이름 생성
        run_dir = base_dir / f"{run_name}_{max_num + 1}"
    
    # 디렉토리 생성
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def evaluate_model_with_memory_management(model, val_loader, device, epoch, save_dir, val_data_path):
    """메모리 관리가 포함된 모델 평가"""
    import gc
    
    # 1. Train 모드에서 eval 모드로 전환
    model.eval()
    
    # 2. 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        # 기존 evaluate_model 로직 실행
        return evaluate_model(model, val_loader, device, epoch, save_dir, val_data_path)
    finally:
        # 3. Validation 후 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()
        model.train()  # 다시 train 모드로

def main():
    # 명령행 인자와 설정 파싱
    args, config = parse_args()
    
    # 디바이스 설정
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 실행 디렉토리 설정
    run_dir = get_run_dir(args, args.model, args.labeled_ratio)
    
    # 로거 설정
    logger = setup_logger(run_dir)
    logger.info(f"Results will be saved to: {run_dir}")
    
    # MC Dropout이 적용된 YOLO 모델 생성
    ema_decay = config.get('model', {}).get('ema', {}).get('decay', 0.999)  # EMA 설정 읽기
    model = YOLOWithMCDropout(
        model_name=args.model,
        dropout_rate=args.dropout_rate,
        feature_alignment_enabled=args.feature_alignment_enabled,
        num_classes=config['data']['nc'],
        ema_decay=ema_decay
    ).to(args.device)
    
    logger.info(f"🔄 EMA Teacher 업데이트 활성화: decay={ema_decay}")
    
    # MC Dropout 탐지기 생성 (결과 저장 디렉토리 포함)
    mc_dropout_save_dir = run_dir / "mcdropout"
    detector = MCDropoutDetector(
        model=model,
        num_samples=args.num_samples,
        dropout_rate=args.dropout_rate,
        box_std_threshold=args.box_std_threshold,
        entropy_threshold=args.entropy_threshold,
        save_dir=mc_dropout_save_dir
    )
    
    # MC Loss 초기화
    mc_loss_fn = MCLoss(
        alpha=config['training']['mc_loss']['alpha'],           # 1.0 (Epistemic)
        beta=config['training']['mc_loss']['beta'],             # 0.5 (Variance)  
        gamma=config['training']['mc_loss']['gamma'],           # 0.3 (Entropy)
        temperature=config['training']['mc_loss']['temperature'], # 1.0
        adaptive_weighting=config['training']['mc_loss']['adaptive_weighting'] # True
    ).to(args.device)
    
    # 옵티마이저 설정
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        momentum=config['training']['optimizer']['momentum'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # 체크포인트에서 재시작
    start_epoch = 0
    best_map = 0.0
    best_map_95 = 0.0  # best_map_95 초기화 추가
    if args.resume:
        model, optimizer, start_epoch, best_map = load_checkpoint(
            model, optimizer, Path(args.resume), scheduler
        )
        # 체크포인트에서 best_map_95도 로드하도록 시도
        try:
            checkpoint = torch.load(Path(args.resume), map_location='cpu')
            if 'best_mAP50_95' in checkpoint:
                best_map_95 = checkpoint['best_mAP50_95']
            elif 'best_map_95' in checkpoint:
                best_map_95 = checkpoint['best_map_95']
        except:
            best_map_95 = 0.0  # 로드 실패 시 기본값
        logger.info(f"Resumed from checkpoint: {args.resume}")
        logger.info(f"Best mAP@0.5: {best_map:.4f}, Best mAP@0.5:0.95: {best_map_95:.4f}")
    
    # 데이터셋 및 데이터 로더 초기화
    # 데이터셋에서는 기본 변환만 적용 (augmentation은 학습 루프에서 동적 적용)
    base_transform = get_val_transform(img_size=config['data']['img_size'])
    val_transform = get_val_transform(img_size=config['data']['img_size'])
    
    logger.info("=== 데이터셋 초기화 시작 ===")
    logger.info(f"Labeled 비율: {args.labeled_ratio}%, 시드: {args.seed}")
    if args.max_samples:
        logger.info(f"⚠️  테스트 모드: 최대 {args.max_samples}개 샘플 사용")
    
    try:
        dataset_manager = SemiSupervisedDataset(
            config_path=args.config,
            percent=args.labeled_ratio,
            seed=args.seed,
            transform=base_transform,  # 기본 변환만 적용
            max_samples=args.max_samples
        )
        
        labeled_loader, unlabeled_loader, val_loader = dataset_manager.get_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # 데이터셋 정보 로깅
        logger.info(f"✓ Labeled 데이터: {len(labeled_loader.dataset)} 샘플, {len(labeled_loader)} 배치")
        logger.info(f"✓ Unlabeled 데이터: {len(unlabeled_loader.dataset)} 샘플, {len(unlabeled_loader)} 배치") 
        logger.info(f"✓ Validation 데이터: {len(val_loader.dataset)} 샘플, {len(val_loader)} 배치")
        
        # 샘플 배치 테스트
        logger.info("=== 데이터 로더 샘플 테스트 ===")
        try:
            sample_batch = next(iter(labeled_loader))
            logger.info(f"✓ Labeled 배치 테스트 성공: 이미지 {sample_batch['images'].shape}, 레이블 {len(sample_batch['labels'])}개")
            
            # 레이블 유효성 검증
            valid_labels = 0
            total_objects = 0
            for labels in sample_batch['labels']:
                if len(labels) > 0:
                    valid_labels += 1
                    total_objects += len(labels)
            
            logger.info(f"  - 유효한 레이블: {valid_labels}/{len(sample_batch['labels'])} 이미지")
            logger.info(f"  - 총 객체 수: {total_objects}")
            
        except Exception as e:
            logger.error(f"❌ Labeled 배치 테스트 실패: {e}")
            raise
            
        try:
            sample_unlabeled = next(iter(unlabeled_loader))
            logger.info(f"✓ Unlabeled 배치 테스트 성공: 이미지 {sample_unlabeled['images'].shape}")
        except Exception as e:
            logger.error(f"❌ Unlabeled 배치 테스트 실패: {e}")
            raise
            
        logger.info("=== 데이터셋 초기화 완료 ===")
        
        # GT 데이터 시각화 수행
        logger.info("=== GT 데이터 시각화 시작 ===")
        try:
            # 클래스 이름 가져오기
            class_names = dataset_manager.class_names
            
            # Train GT 데이터 시각화
            logger.info("Train GT 데이터 시각화 중...")
            train_gt_dir = run_dir
            visualize_gt_data(
                data_loader=labeled_loader,
                class_names=class_names,
                save_dir=train_gt_dir,
                data_type="train",
                max_batches=3,
                max_images_per_batch=8
            )
            
            # Val GT 데이터 시각화
            logger.info("Validation GT 데이터 시각화 중...")
            val_gt_dir = run_dir
            visualize_gt_data(
                data_loader=val_loader,
                class_names=class_names,
                save_dir=val_gt_dir,
                data_type="val",
                max_batches=2,
                max_images_per_batch=8
            )
            
            logger.info(f"✓ GT 데이터 시각화 완료: {run_dir / 'gt_visualization'}")
            
        except Exception as e:
            logger.warning(f"GT 데이터 시각화 실패: {e}")
            logger.warning("학습은 계속 진행됩니다.")
        
        logger.info("=== GT 데이터 시각화 완료 ===")
        
    except Exception as e:
        logger.error(f"❌ 데이터셋 초기화 실패: {e}")
        logger.error("다음 사항을 확인하세요:")
        logger.error("1. COCO 데이터셋 경로: /media/lee/Data/COCO/train2017/")
        logger.error("2. 이미지 디렉토리: /media/lee/Data/COCO/train2017/images/")
        logger.error("3. 레이블 디렉토리: /media/lee/Data/COCO/train2017/labels/")
        logger.error("4. 데이터 리스트 파일들이 올바른 경로를 가리키는지 확인")
        
        # 더 상세한 오류 정보 출력
        import traceback
        logger.error("상세 오류 정보:")
        logger.error(traceback.format_exc())
        raise
    
    # 메트릭 기록
    metrics = {
        'epoch': [],
        'loss': [],
        'mAP50': [],
        'mAP50-95': []
    }
    
    # 학습 루프
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n🔄 Starting Epoch {epoch+1}/{args.epochs}")
        
        # 학습 단계
        avg_loss = train_one_epoch(
            model=model,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            optimizer=optimizer,
            detector=detector,
            mc_loss_fn=mc_loss_fn,  # MC Loss 함수 추가
            device=args.device,
            epoch=epoch,
            logger=logger,
            config=config,
            pseudo_label_weight=args.pseudo_label_weight,
            pseudo_label_start_epoch=args.pseudo_label_start_epoch,
            conf_threshold=args.conf_threshold,
            uncertainty_weight=args.uncertainty_weight,
            alignment_weight=args.feature_alignment_weight if args.feature_alignment_enabled else 0.0
        )
        
        logger.info(f"📊 Epoch {epoch+1} - Average Loss: {avg_loss:.6f}")
        
        # 스케줄러 업데이트
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"📈 Learning Rate: {current_lr:.8f}")
        
        # 에포크별 저장 및 평가
        is_best = False
        
        # 검증 및 체크포인트 저장 (10 에포크마다 또는 마지막 에포크)
        if (epoch + 1) % config['training']['save_interval'] == 0 or epoch == config['training']['epochs'] - 1:
            logger.info(f"💾 Saving checkpoint and evaluating at epoch {epoch+1}")
            
            # MC Dropout 품질 트렌드 분석 및 저장 (10 에포크마다)
            if (epoch + 1) % 10 == 0:
                save_mc_dropout_analysis(
                    model=model,
                    detector=detector,
                    config=config,
                    epoch=epoch,
                    save_dir=run_dir,
                    logger=logger,
                    data_loader=labeled_loader  # 실제 학습 데이터 로더 전달
                )
            
            # 검증 수행
            try:
                logger.info(f"🔍 Evaluating model at epoch {epoch+1}")
                mAP50, mAP50_95 = evaluate_model_with_memory_management(
                    model=model,
                    val_loader=val_loader,
                    device=args.device,
                    epoch=epoch,
                    save_dir=run_dir,
                    val_data_path=args.val_data_path
                )
                
                logger.info(f"📊 Epoch {epoch+1} - mAP@0.5: {mAP50:.4f}, mAP@0.5:0.95: {mAP50_95:.4f}")
                
                # 최고 성능 체크
                if mAP50 > best_map:
                    best_map = mAP50
                    best_map_95 = mAP50_95
                    is_best = True
                    logger.info(f"🎉 New best mAP@0.5: {best_map:.4f}")
                    
            except Exception as eval_error:
                logger.error(f"❌ Evaluation failed at epoch {epoch+1}: {eval_error}")
                mAP50, mAP50_95 = 0.0, 0.0
            
            # 체크포인트 저장
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'mAP50': mAP50,
                'mAP50_95': mAP50_95,
                'best_mAP50': best_map,
                'best_mAP50_95': best_map_95,
                'config': config
            }
            
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            # 최신 체크포인트 저장
            latest_path = run_dir / 'latest_model.pt'
            torch.save(checkpoint, latest_path)
            logger.info(f"💾 Latest checkpoint saved: {latest_path}")
            
            # 최고 성능 모델 저장
            if is_best:
                best_path = run_dir / 'best_model.pt'
                torch.save(checkpoint, best_path)
                logger.info(f"🏆 Best model saved: {best_path}")
            
            # 정기적 백업 (50 에포크마다)
            if (epoch + 1) % 50 == 0:
                backup_path = run_dir / f'model_epoch_{epoch+1}.pt'
                torch.save(checkpoint, backup_path)
                logger.info(f"📦 Backup saved: {backup_path}")
    
    # 최종 평가 수행
    logger.info("Performing final evaluation...")
    final_mAP50, final_mAP50_95 = evaluate_model_with_memory_management(
        model=model,
        val_loader=val_loader,
        device=args.device,
        epoch=args.epochs-1,
        save_dir=run_dir,
        val_data_path=args.val_data_path
    )
    
    # 최종 메트릭 기록
    metrics['mAP50'].append(final_mAP50)
    metrics['mAP50-95'].append(final_mAP50_95)
    
    # 최종 메트릭 시각화
    plot_metrics(
        metrics=metrics,
        save_path=run_dir / 'final_metrics.png'
    )
    
    # 최종 결과 로깅
    logger.info(
        f"Training completed - "
        f"Final mAP50: {final_mAP50:.4f}, "
        f"Final mAP50-95: {final_mAP50_95:.4f}"
    )
    
    # MC Dropout 전체 실험 요약 보고서 생성 - 주석처리: 컴퓨터 멈춤 방지
    # logger.info("Generating MC Dropout experiment summary report...")
    # detector.create_summary_report()
    # logger.info(f"MC Dropout experiment summary saved to {mc_dropout_save_dir}")
    
    # 최종 모델 저장
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=args.epochs,
        save_path=run_dir / 'final_model.pth',
        scheduler=scheduler,
        best_map=final_mAP50
    )

def save_mc_dropout_analysis(
    model: torch.nn.Module,
    detector: MCDropoutDetector,
    config: dict,
    epoch: int,
    save_dir: Path,
    logger: logging.Logger,
    data_loader: torch.utils.data.DataLoader = None
):
    """학습 중간 MC Dropout 품질 분석 결과 저장
    
    Args:
        model: 분석할 모델
        detector: MC Dropout 탐지기
        config: 설정
        epoch: 현재 에포크
        save_dir: 저장 디렉토리
        logger: 로거
    """
    try:
        from datetime import datetime
        
        # 분석 파일 경로
        analysis_file = save_dir / f"mc_dropout_training_analysis_epoch_{epoch+1}.txt"
        
        # 실제 데이터 사용 (더미 데이터 대신)
        import torch
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
            f.write("=" * 80 + "\n")
            f.write(f"MC DROPOUT 학습 중간 분석 보고서 - Epoch {epoch+1}\n")
            f.write("=" * 80 + "\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"학습 에포크: {epoch+1}/{config['training']['epochs']}\n")
            f.write(f"현재 설정 num_samples: {config['model']['dropout']['num_samples']}\n")
            f.write(f"현재 설정 dropout_rate: {config['model']['dropout']['rate']}\n")
            f.write("=" * 80 + "\n\n")
            
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
                        uncertainty_results = temp_detector.predict_with_uncertainty(test_batch)
                    
                    inference_time = time.time() - start_time
                    
                    # 결과 분석
                    detection_count = len(uncertainty_results) if uncertainty_results else 0
                    
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
                    
                    # 결과 출력
                    f.write(f"{num_samples:<8} "
                           f"{inference_time:<10.3f} "
                           f"{detection_count:<12} "
                           f"{avg_box_var:<12.4f} "
                           f"{avg_confidence:<12.4f}\n")
                    
                except Exception as e:
                    f.write(f"{num_samples:<8} ERROR: {str(e)[:30]}\n")
            
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
        
        logger.info(f"📁 MC Dropout 분석 결과 저장: {analysis_file}")
        
    except Exception as e:
        logger.warning(f"MC Dropout 분석 저장 실패: {e}")

def update_pseudo_labels(model, unlabeled_loader, detector, conf_threshold, device, config):
    """
    Teacher MC Dropout으로 Weak Augmentation된 unlabeled 데이터에서 고품질 pseudo label 생성
    
    연구 시나리오:
    - Teacher 모델: Weak Augmentation으로 안정적인 pseudo label 생성
    - MC Dropout 불확실성 추정으로 고품질 pseudo label 필터링
    - 예측 분산이 낮을수록 신뢰도 높은 detection으로 판정
    """
    model.eval()  # Teacher는 eval 모드에서 MC Dropout
    pseudo_labels = []
    
    # Teacher용 Weak Augmentation
    weak_transform = get_weak_augmentation(config['data']['img_size'])
    
    # MC Dropout 설정
    num_mc_samples = config['model']['dropout']['num_samples']
    box_std_threshold = config['training']['semi_supervised']['uncertainty']['box_std_threshold']
    entropy_threshold = config['training']['semi_supervised']['uncertainty']['entropy_threshold']
    
    print(f"🔍 Teacher MC Dropout Pseudo Label 생성 시작 (MC samples: {num_mc_samples})")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(unlabeled_loader):
            if batch_idx >= 10:  # 테스트용으로 제한
                break
                
            # === Teacher MC Dropout: Weak Augmentation으로 안정적인 예측 ===
            weak_images = []
            for img in batch['images']:
                if isinstance(img, torch.Tensor):
                    img_pil = tensor_to_pil(img)
                else:
                    img_pil = img
                
                # Teacher용 Weak augmentation 적용
                weak_img = weak_transform(img_pil)
                weak_images.append(weak_img)
            
            weak_images = torch.stack(weak_images).to(device)
            
            # Teacher MC Dropout 단일 호출 (내부적으로 num_samples만큼 MC 샘플링 수행)
            try:
                model.eval()  # Teacher는 eval 모드에서 MC Dropout
                with torch.no_grad():
                    # === 개선된 단일 MC Dropout 호출 ===
                    # predict_with_uncertainty 함수가 내부적으로 이미 num_samples만큼 MC 샘플링을 수행함
                    mc_result = detector.predict_with_uncertainty(weak_images, device=device)
                
                if mc_result:
                    # 기존 코드와의 호환성을 위해 리스트로 래핑
                    mc_results_list = [mc_result]
                    
                    print(f"  📊 MC Dropout completed: {len(mc_result)} image results with {detector.num_samples} internal samples each")
                    
                    if len(mc_results_list) >= 1:  # 최소 1개 결과 필요
                        # === MC Dropout 불확실성 기반 고품질 Pseudo Label 선별 ===
                        
                        # 각 detection에 대해 MC 샘플들의 분산 계산
                        high_quality_pseudo_labels = []
                        
                        for img_idx in range(len(weak_images)):
                            # 해당 이미지의 모든 MC 샘플 수집
                            img_mc_detections = []
                            
                            for mc_results in mc_results_list:
                                if img_idx < len(mc_results):
                                    result = mc_results[img_idx]
                                    if len(result.get('boxes', [])) > 0:
                                        img_mc_detections.append(result)
                            
                            # MC Dropout에서 한 번의 호출로도 내부적으로 여러 샘플을 비교하므로
                            # 최소 조건을 1개로 낮춤
                            if len(img_mc_detections) >= 1:  
                                # Detection들의 일관성 분석 (단일 결과라도 불확실성 지표 활용)
                                consistent_detections = extract_confident_detections(
                                    img_mc_detections[0],  # 첫 번째 (유일한) MC 결과 사용
                                    box_std_threshold=box_std_threshold,
                                    entropy_threshold=entropy_threshold,
                                    conf_threshold=conf_threshold
                                )
                                
                                if len(consistent_detections) > 0:
                                    high_quality_pseudo_labels.append({
                                        'boxes': consistent_detections,
                                        'image_path': batch.get('paths', [''])[img_idx] if 'paths' in batch and img_idx < len(batch.get('paths', [])) else '',
                                        'uncertainty_stats': {
                                            'mc_samples': detector.num_samples,  # 내부 MC 샘플 수
                                            'detections_count': len(consistent_detections)
                                        }
                                    })
                        
                        pseudo_labels.extend(high_quality_pseudo_labels)
                        
                        if high_quality_pseudo_labels:
                            total_detections = sum(len(pl['boxes']) for pl in high_quality_pseudo_labels)
                            print(f"  📊 Batch {batch_idx}: {len(high_quality_pseudo_labels)} images, {total_detections} high-quality detections")
                    
            except Exception as e:
                print(f"❌ Error in Teacher MC Dropout pseudo label generation (batch {batch_idx}): {e}")
                continue
    
    print(f"✅ Teacher MC Dropout Pseudo Label 생성 완료: {len(pseudo_labels)} 고품질 pseudo labels")
    return pseudo_labels

def extract_confident_detections(mc_result, box_std_threshold=0.1, entropy_threshold=0.5, conf_threshold=0.5):
    """
    MC Dropout 결과에서 불확실성 지표를 활용하여 고신뢰도 detection 추출
    
    Args:
        mc_result: predict_with_uncertainty의 단일 이미지 결과
        box_std_threshold: Box 좌표 분산 임계값
        entropy_threshold: 클래스 엔트로피 임계값  
        conf_threshold: 신뢰도 임계값
    
    Returns:
        고신뢰도 detection들의 YOLO 형식 텐서 [class_id, x, y, w, h]
    """
    try:
        if not mc_result or len(mc_result.get('boxes', [])) == 0:
            return torch.empty(0, 5)
        
        boxes = mc_result['boxes']
        scores = mc_result['scores'] 
        labels = mc_result['labels']
        box_std = mc_result.get('box_std', torch.zeros_like(boxes))
        class_entropy = mc_result.get('class_entropy', torch.zeros(len(boxes)))
        
        # 다중 조건 필터링
        confidence_mask = scores > conf_threshold
        box_uncertainty_mask = box_std.mean(dim=-1) < box_std_threshold  # 박스 좌표 분산이 낮음
        class_uncertainty_mask = class_entropy < entropy_threshold  # 클래스 엔트로피가 낮음
        
        # 모든 조건을 만족하는 detection만 선택
        final_mask = confidence_mask & box_uncertainty_mask & class_uncertainty_mask
        
        if final_mask.sum() == 0:
            return torch.empty(0, 5)
        
        # 필터링된 결과 추출
        filtered_boxes = boxes[final_mask]
        filtered_labels = labels[final_mask]
        
        # YOLO 형식으로 변환 [class_id, x, y, w, h]
        confident_detections = []
        for i in range(len(filtered_boxes)):
            yolo_detection = torch.zeros(5)
            yolo_detection[0] = filtered_labels[i].float()  # class_id
            yolo_detection[1:5] = filtered_boxes[i]  # x, y, w, h
            confident_detections.append(yolo_detection)
        
        if confident_detections:
            return torch.stack(confident_detections)
        else:
            return torch.empty(0, 5)
            
    except Exception as e:
        print(f"❌ Error in extract_confident_detections: {e}")
        return torch.empty(0, 5)

def analyze_detection_consistency(mc_detections, box_std_threshold=0.1, entropy_threshold=0.5, conf_threshold=0.5):
    """
    MC Dropout 샘플들의 detection 일관성 분석하여 고품질 pseudo label 선별
    
    Args:
        mc_detections: MC Dropout 샘플들의 detection 결과 리스트
        box_std_threshold: Box 좌표 분산 임계값 (낮을수록 일관성 높음)
        entropy_threshold: 클래스 예측 엔트로피 임계값 (낮을수록 확신도 높음)
        conf_threshold: 신뢰도 임계값
    
    Returns:
        일관성 높은 detection들의 YOLO 형식 텐서
    """
    if not mc_detections or len(mc_detections) < 2:
        return torch.empty(0, 5)
    
    try:
        # 모든 MC 샘플의 detection 수집
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for detection in mc_detections:
            if 'boxes' in detection and len(detection['boxes']) > 0:
                all_boxes.append(detection['boxes'].cpu())
                all_scores.append(detection['scores'].cpu())
                all_labels.append(detection['labels'].cpu())
        
        if not all_boxes:
            return torch.empty(0, 5)
        
        # Detection 클러스터링 (같은 객체에 대한 여러 예측들 그룹화)
        consistent_detections = []
        
        # 간단한 클러스터링: 첫 번째 샘플의 각 detection에 대해
        # 다른 샘플들에서 유사한 위치의 detection들 찾기
        first_boxes = all_boxes[0]
        first_scores = all_scores[0]
        first_labels = all_labels[0]
        
        for det_idx in range(len(first_boxes)):
            if first_scores[det_idx] < conf_threshold:
                continue
                
            box = first_boxes[det_idx]
            score = first_scores[det_idx]
            label = first_labels[det_idx]
            
            # 다른 MC 샘플들에서 유사한 detection 찾기
            similar_detections = [(box, score, label)]
            
            for mc_idx in range(1, len(all_boxes)):
                mc_boxes = all_boxes[mc_idx]
                mc_scores = all_scores[mc_idx]
                mc_labels = all_labels[mc_idx]
                
                # IoU 기반으로 유사한 detection 찾기
                for mc_det_idx in range(len(mc_boxes)):
                    if mc_scores[mc_det_idx] < conf_threshold:
                        continue
                        
                    mc_box = mc_boxes[mc_det_idx]
                    mc_score = mc_scores[mc_det_idx]
                    mc_label = mc_labels[mc_det_idx]
                    
                    # 같은 클래스이고 IoU가 높은 경우
                    if mc_label == label:
                        iou = calculate_iou(box, mc_box)
                        if iou > 0.5:  # IoU threshold
                            similar_detections.append((mc_box, mc_score, mc_label))
                            break
            
            # 충분한 일관성이 있는 경우만 선택
            if len(similar_detections) >= max(2, len(mc_detections) // 2):
                # Box 좌표의 분산 계산
                boxes_tensor = torch.stack([det[0] for det in similar_detections])
                box_variance = torch.var(boxes_tensor, dim=0).mean()
                
                # 신뢰도 점수의 분산 계산
                scores_tensor = torch.stack([det[1] for det in similar_detections])
                score_variance = torch.var(scores_tensor)
                
                # 일관성 기준 검사
                if box_variance < box_std_threshold and score_variance < 0.1:
                    # 평균 box와 평균 score 계산
                    mean_box = torch.mean(boxes_tensor, dim=0)
                    mean_score = torch.mean(scores_tensor)
                    
                    # YOLO 형식으로 변환 [class_id, x, y, w, h]
                    yolo_detection = torch.zeros(5)
                    yolo_detection[0] = label.float()
                    yolo_detection[1:] = mean_box
                    
                    consistent_detections.append(yolo_detection)
        
        if consistent_detections:
            return torch.stack(consistent_detections)
        else:
            return torch.empty(0, 5)
            
    except Exception as e:
        print(f"❌ Error in detection consistency analysis: {e}")
        return torch.empty(0, 5)

def calculate_iou(box1, box2):
    """
    두 bounding box의 IoU 계산 (YOLO 형식: [x, y, w, h])
    """
    try:
        # Center format to corner format
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Union
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
        
    except Exception:
        return 0.0

if __name__ == "__main__":
    main()