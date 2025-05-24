from pathlib import Path
import argparse
import torch
import torch.optim as optim
import logging
from models.yolo_mc import YOLOWithMCDropout
from uncertainty.mc_dropout import MCDropoutDetector
from data.semi_supervised_dataset import SemiSupervisedDataset
from utils.train_utils import (
    setup_logger,
    save_checkpoint,
    load_checkpoint,
    update_pseudo_labels,
    compute_loss,
    evaluate_model
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
from tqdm import tqdm
from utils.visualization import visualize_batch, plot_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO with MC Dropout for Semi-Supervised Learning')
    
    # 기본 설정
    parser.add_argument('--config', type=str, 
                      default=str(Path(__file__).parent / 'configs' / 'yolo_config.yaml'),
                      help='YAML 설정 파일 경로')
    
    # 모델 설정
    parser.add_argument('--model', type=str, default='yolov8m',
                      help='YOLO 모델 이름 (yolov8n, yolov8s, yolov8m 등)')
    parser.add_argument('--dropout-rate', type=float, default=0.1,
                      help='MC Dropout 비율')
    parser.add_argument('--num-samples', type=int, default=10,
                      help='MC Dropout 샘플링 횟수')
    
    # 데이터 설정
    parser.add_argument('--labeled-ratio', type=float, default=10,
                      help='레이블이 있는 데이터의 비율 (1, 2, 5, 10)')
    parser.add_argument('--max-samples', type=int, default=None,
                      help='학습에 사용할 최대 샘플 수 (테스트용, 기본값: None은 전체 데이터 사용)')
    
    # 학습 설정
    parser.add_argument('--batch-size', type=int, default=32,
                      help='배치 크기')
    parser.add_argument('--epochs', type=int, default=100,
                      help='학습 에포크 수')
    parser.add_argument('--device', type=str, default=None,
                      help='실행 디바이스 (예: cpu, cuda:0)')
    parser.add_argument('--resume', type=str, default=None,
                      help='체크포인트에서 재시작')
    parser.add_argument('--val-interval', type=int, default=1,
                      help='검증을 수행할 에포크 간격')
    parser.add_argument('--save-interval', type=int, default=10,
                      help='체크포인트를 저장할 에포크 간격')
    
    # 불확실성 임계값
    parser.add_argument('--box-std-threshold', type=float, default=0.1,
                      help='박스 좌표 표준편차 임계값')
    parser.add_argument('--entropy-threshold', type=float, default=0.5,
                      help='클래스 엔트로피 임계값')
    
    
    return parser.parse_args()

def get_transform(train: bool = True, img_size: int = 640):
    """데이터 변환 함수 생성"""
    if train:
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.RandomCrop(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    else:
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    return transform

def train_one_epoch(
    model: torch.nn.Module,
    labeled_loader: torch.utils.data.DataLoader,
    unlabeled_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    detector: MCDropoutDetector,
    device: str,
    epoch: int,
    logger: logging.Logger,
    config: dict,
    pseudo_label_weight: float = 0.5,
    pseudo_label_start_epoch: int = 10,
    conf_threshold: float = 0.5,
    uncertainty_weight: float = 50.0
):
    """한 에포크 학습"""
    model.train()
    total_loss = 0
    labeled_iter = iter(labeled_loader)
    
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
    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}", dynamic_ncols=True)
    for batch_idx in pbar:
        # 레이블된 데이터로부터 배치 가져오기
        try:
            labeled_batch = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            labeled_batch = next(labeled_iter)
        
        # 레이블된 데이터 학습
        labeled_images = labeled_batch['images'].to(device)
        
        # YOLO 형식으로 targets 변환
        labeled_targets = []
        for i, labels in enumerate(labeled_batch['labels']):
            if len(labels) > 0:
                batch_labels = torch.zeros((len(labels), 6), device=device)
                batch_labels[:, 0] = i  # batch index
                batch_labels[:, 1:] = labels  # [class_id, x_center, y_center, width, height]
                labeled_targets.append(batch_labels)
        
        if not labeled_targets:  # 빈 배치 처리
            labeled_targets = torch.zeros((0, 6), device=device)
        else:
            labeled_targets = torch.cat(labeled_targets, dim=0)
        
        predictions = model(labeled_images)
        labeled_loss, loss_dict = compute_loss(predictions, labeled_targets, model)
        
        # 레이블되지 않은 데이터 학습 (의사 레이블 사용)
        if epoch >= pseudo_label_start_epoch and pseudo_labels:
            unlabeled_batch = next(iter(unlabeled_loader))
            unlabeled_images = unlabeled_batch['images'].to(device)
            
            # MC Dropout을 통한 불확실성 추정
            features_list = []  # 각 feature map별로 예측 결과 저장
            model.train()  # MC Dropout 활성화
            with torch.no_grad():
                for _ in range(detector.num_samples):
                    pred = model(unlabeled_images)
                    if isinstance(pred, list):
                        # 각 feature map별로 저장
                        if not features_list:
                            features_list = [[] for _ in range(len(pred))]
                        for i, feat in enumerate(pred):
                            features_list[i].append(feat)
                    else:
                        features_list.append(pred)
            
            # 각 feature map별로 평균과 분산 계산
            mean_features = []
            variance_features = []
            for feature_samples in features_list:
                # (num_samples, batch, anchors, grid_h, grid_w, channels)
                feature_stack = torch.stack(feature_samples)
                mean_features.append(torch.mean(feature_stack, dim=0))
                variance_features.append(torch.std(feature_stack, dim=0) ** 2)
            
            # 전체 분산의 평균 계산
            total_variance = torch.mean(torch.stack([v.mean() for v in variance_features]))
            
            # 의사 레이블을 YOLO 형식으로 변환
            pseudo_targets = []
            for i, p in enumerate(pseudo_labels):
                if len(p['boxes']) > 0:
                    batch_labels = torch.zeros((len(p['boxes']), 6), device=device)
                    batch_labels[:, 0] = i  # batch index
                    batch_labels[:, 1:] = torch.tensor(p['boxes'], device=device)
                    pseudo_targets.append(batch_labels)
            
            if pseudo_targets:
                pseudo_targets = torch.cat(pseudo_targets, dim=0)
                pseudo_loss, pseudo_loss_dict = compute_loss(
                    {'features': mean_features},  # 평균 예측 사용
                    pseudo_targets,
                    model,
                    uncertainty=total_variance,  # 전체 예측의 평균 분산 사용
                    alpha=uncertainty_weight
                )
                
                # 전체 손실 계산
                total_loss = labeled_loss + pseudo_label_weight * pseudo_loss
                
                # 로깅
                loss_dict.update({f'pseudo_{k}': v for k, v in pseudo_loss_dict.items()})
                loss_dict['uncertainty'] = total_variance.item()
            else:
                total_loss = labeled_loss
        else:
            total_loss = labeled_loss
        
        # 역전파
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 현재 배치의 loss 값을 tqdm description에 업데이트
        desc = f"Epoch {epoch}"
        if loss_dict:
            desc += f" - Loss: {total_loss.item():.4f}"
            for k, v in loss_dict.items():
                desc += f", {k}: {v:.4f}"
        pbar.set_description(desc)
    
    return total_loss.item()

def get_run_dir(model_name: str, labeled_ratio: float) -> Path:
    """실행 디렉토리 생성 및 반환
    
    Args:
        model_name: YOLO 모델 이름 (e.g., "yolov8m")
        labeled_ratio: 레이블된 데이터 비율
    
    Returns:
        실행 디렉토리 경로
    """
    # runs/train 디렉토리 생성
    base_dir = Path("runs") / "train"
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

def evaluate_model(model, val_loader, device, epoch, save_dir):
    """모델 성능 평가
    
    Args:
        model: 평가할 모델
        val_loader: 검증 데이터 로더
        device: 실행 디바이스
        epoch: 현재 에포크
        save_dir: 결과 저장 디렉토리
    
    Returns:
        mAP50, mAP50-95 값
    """
    model.eval()
    results = []
    
    # 첫 번째 배치에 대해서만 시각화 수행
    visualized = False
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Evaluating epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(device)
            targets = batch['labels']
            
            # 예측 수행
            predictions = model(images)
            
            # 첫 번째 배치 시각화
            if not visualized:
                vis_path = save_dir / f'val_visualization_epoch_{epoch}.png'
                visualize_batch(
                    images=images,
                    targets=targets,
                    predictions=predictions,
                    class_names=val_loader.dataset.class_names,
                    save_path=vis_path
                )
                visualized = True
            
            # YOLO 모델의 내장 평가 함수 사용
            results.extend(predictions)
    
    # YOLO 모델의 내장 평가 함수로 mAP 계산
    metrics = model.model.val(val_loader)
    mAP50 = metrics.results_dict.get('metrics/mAP50(B)', 0.0)
    mAP50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', 0.0)
    
    # 평가 결과를 txt 파일로 저장
    results_file = save_dir / f'eval_results_epoch_{epoch}.txt'
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results for Epoch {epoch}\n")
        f.write("=" * 50 + "\n\n")
        
        # 기본 메트릭 저장
        f.write(f"mAP50: {mAP50:.4f}\n")
        f.write(f"mAP50-95: {mAP50_95:.4f}\n\n")
        
        # 클래스별 메트릭 저장
        f.write("Class-wise Results:\n")
        f.write("-" * 30 + "\n")
        for cls_name, cls_metrics in metrics.results_dict.items():
            if cls_name.startswith('metrics/'):
                continue
            f.write(f"{cls_name}: {cls_metrics:.4f}\n")
        
        # 추가 메트릭 저장
        f.write("\nAdditional Metrics:\n")
        f.write("-" * 30 + "\n")
        for metric_name, metric_value in metrics.results_dict.items():
            if metric_name.startswith('metrics/'):
                f.write(f"{metric_name}: {metric_value:.4f}\n")
    
    return mAP50, mAP50_95

def main():
    # 명령행 인자 파싱
    args = parse_args()
    
    # 설정 파일 로드
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # 설정값 정수형 변환 및 변수 참조 처리
    if isinstance(config['training']['semi_supervised']['pseudo_label_start_epoch'], str):
        if config['training']['semi_supervised']['pseudo_label_start_epoch'] == '${training.mature_epoch}':
            config['training']['semi_supervised']['pseudo_label_start_epoch'] = config['training']['mature_epoch']
    
    # 디바이스 설정
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 실행 디렉토리 설정
    run_dir = get_run_dir(args.model, args.labeled_ratio)
    
    # 로거 설정
    logger = setup_logger(run_dir)
    logger.info(f"Results will be saved to: {run_dir}")
    
    # MC Dropout이 적용된 YOLO 모델 생성
    model = YOLOWithMCDropout(
        model_name=args.model,
        dropout_rate=args.dropout_rate
    ).to(args.device)
    
    # MC Dropout 탐지기 생성
    detector = MCDropoutDetector(
        model=model,
        num_samples=args.num_samples,
        dropout_rate=args.dropout_rate,
        box_std_threshold=args.box_std_threshold,
        entropy_threshold=args.entropy_threshold
    )
    
    # 옵티마이저 설정
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005
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
    if args.resume:
        model, optimizer, start_epoch, best_map = load_checkpoint(
            model, optimizer, Path(args.resume), scheduler
        )
        logger.info(f"Resumed from checkpoint: {args.resume}")
    
    # 데이터셋 및 데이터 로더 초기화
    train_transform = get_transform(train=True)
    val_transform = get_transform(train=False)
    
    dataset_manager = SemiSupervisedDataset(
        config_path=args.config,
        percent=args.labeled_ratio,
        transform=train_transform,
        max_samples=args.max_samples
    )
    
    labeled_loader, unlabeled_loader, val_loader = dataset_manager.get_dataloaders(
        batch_size=args.batch_size
    )
    
    # 테스트 모드일 경우 로그 출력
    if args.max_samples:
        logger.info(f"테스트 모드: {args.max_samples}개의 샘플만 사용하여 학습을 진행합니다.")
    
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
        # 한 에포크 학습
        train_loss = train_one_epoch(
            model=model,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            optimizer=optimizer,
            detector=detector,
            device=args.device,
            epoch=epoch,
            logger=logger,
            config=config,
            pseudo_label_weight=config['training']['semi_supervised']['pseudo_label_weight'],
            pseudo_label_start_epoch=config['training']['semi_supervised']['pseudo_label_start_epoch'],
            conf_threshold=config['training']['semi_supervised']['conf_threshold'],
            uncertainty_weight=config['training']['uncertainty_weight']
        )
        
        # 학습률 업데이트
        scheduler.step()
        
        # 메트릭 기록
        metrics['epoch'].append(epoch)
        metrics['loss'].append(train_loss)
        
        # 로깅
        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Loss: {train_loss:.4f}")
        
        # 체크포인트 저장
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch + 1,
                run_dir / f'checkpoint_epoch_{epoch + 1}.pth',
                scheduler, best_map
            )
    
    # 최종 평가 수행
    logger.info("Performing final evaluation...")
    final_mAP50, final_mAP50_95 = evaluate_model(
        model=model,
        val_loader=val_loader,
        device=args.device,
        epoch=args.epochs-1,
        save_dir=run_dir
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
    
    # 최종 모델 저장
    save_checkpoint(
        model, optimizer, args.epochs,
        run_dir / 'final_model.pth',
        scheduler, final_mAP50
    )

if __name__ == "__main__":
    main() 