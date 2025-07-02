#!/usr/bin/env python3

import os
import sys
import argparse
import torch
import torch.multiprocessing as mp
from pathlib import Path
import yaml
import logging
from typing import Dict, Any

# 현재 디렉토리를 Python 경로에 추가 (UV 환경 호환)
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from utils.distributed_utils import (
    setup_gpu_config,
    setup_distributed_training,
    cleanup_distributed,
    is_main_process,
    get_world_size,
    get_rank,
    setup_model_for_distributed,
    setup_dataloader_for_distributed,
    save_checkpoint_distributed,
    setup_logger_distributed,
    log_gpu_info,
    find_free_port
)

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
from tqdm import tqdm
from utils.visualization import visualize_batch, plot_metrics, visualize_gt_data, plot_training_curves, plot_uncertainty_distribution

# 기존 train.py의 함수들 import
from train import (
    get_transform,
    train_one_epoch,
    evaluate_mc_dropout_quality,
    get_run_dir,
    evaluate_model_with_memory_management,
    save_mc_dropout_analysis,
    update_pseudo_labels,
    denormalize_tensor,
    tensor_to_pil,
    apply_cutout
)


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='YOLO with MC Dropout - Distributed Training')
    
    # 기본 설정
    parser.add_argument('--config', type=str, 
                      default=str(Path(__file__).parent / 'configs' / 'yolo_config.yaml'),
                      help='YAML 설정 파일 경로')
    
    # 분산 학습 관련 설정
    parser.add_argument('--distributed', action='store_true',
                      help='분산 학습 강제 활성화')
    parser.add_argument('--gpu_ids', type=str, default=None,
                      help='사용할 GPU ID (쉼표로 구분, 예: 0,1,2,3)')
    parser.add_argument('--world_size', type=int, default=None,
                      help='전체 프로세스 수')
    parser.add_argument('--rank', type=int, default=None,
                      help='현재 프로세스 순위')
    parser.add_argument('--local_rank', type=int, default=None,
                      help='로컬 프로세스 순위')
    parser.add_argument('--master_addr', type=str, default='localhost',
                      help='마스터 노드 주소')
    parser.add_argument('--master_port', type=str, default=None,
                      help='마스터 노드 포트')
    
    # 기본 실행 관련 설정
    parser.add_argument('--save_dir', type=str, default=".",
                      help='프로젝트 루트 디렉토리')
    parser.add_argument('--val_data_path', type=str, 
                      default="/media/oem/personal_vol/yklee/ssod_yolo_mc/src/configs/coco_val.yaml",
                      help='검증 데이터 경로')
    parser.add_argument('--device', type=str, default=None,
                      help='실행 디바이스')
    parser.add_argument('--resume', type=str, default=None,
                      help='체크포인트에서 재시작')
    
    return parser.parse_args()


def load_config(config_path: str, args) -> Dict[str, Any]:
    """설정 파일 로드 및 명령행 인자로 오버라이드"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 명령행 인자로 설정 오버라이드
    if args.gpu_ids is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        config['gpu']['gpu_ids'] = gpu_ids
        config['gpu']['auto_detect'] = False
    
    if args.distributed:
        config['gpu']['distributed']['enabled'] = True
    
    if args.world_size is not None:
        config['gpu']['distributed']['world_size'] = args.world_size
    
    if args.rank is not None:
        config['gpu']['distributed']['rank'] = args.rank
    
    if args.local_rank is not None:
        config['gpu']['distributed']['local_rank'] = args.local_rank
    
    if args.master_addr:
        config['gpu']['distributed']['master_addr'] = args.master_addr
    
    if args.master_port is not None:
        config['gpu']['distributed']['master_port'] = args.master_port
    
    # 설정값 정수형 변환 및 변수 참조 처리
    if isinstance(config['training']['semi_supervised']['pseudo_label_start_epoch'], str):
        if config['training']['semi_supervised']['pseudo_label_start_epoch'] == '${training.mature_epoch}':
            config['training']['semi_supervised']['pseudo_label_start_epoch'] = config['training']['mature_epoch']
    
    return config


def create_training_objects(config: Dict[str, Any], device: str, rank: int = 0):
    """학습에 필요한 객체들 생성"""
    
    # 데이터 변환 함수
    weak_transform = get_transform(train=True, img_size=config['data']['img_size'], augmentation_type="weak")
    strong_transform = get_transform(train=True, img_size=config['data']['img_size'], augmentation_type="strong")
    val_transform = get_transform(train=False, img_size=config['data']['img_size'])
    
    # 데이터셋 생성 (Teacher: weak aug, Student: strong aug)
    labeled_dataset = SemiSupervisedDataset(
        config_path=Path(__file__).parent / 'configs' / 'yolo_config.yaml',
        percent=config['data']['labeled_ratio'],
        seed=config['data']['seed'],
        transform=strong_transform,  # Student model용 strong augmentation
        max_samples=config['data']['max_samples']
    )
    
    # Teacher용 Weak augmentation 데이터셋 (pseudo label 생성용)
    teacher_dataset = SemiSupervisedDataset(
        config_path=Path(__file__).parent / 'configs' / 'yolo_config.yaml',
        percent=config['data']['labeled_ratio'],
        seed=config['data']['seed'],
        transform=weak_transform,  # Teacher model용 weak augmentation
        max_samples=config['data']['max_samples']
    )
    
    # 모델 생성 (Teacher와 Student 동일한 아키텍처)
    model = YOLOWithMCDropout(
        model_name=config['model']['name'],
        num_classes=config['data']['nc'],
        dropout_rate=config['model']['dropout']['rate'],
        feature_alignment_enabled=config['model']['feature_alignment']['enabled']
    )
    
    # MC Dropout 탐지기
    detector = MCDropoutDetector(
        num_samples=config['model']['dropout']['num_samples'],
        box_std_threshold=config['training']['semi_supervised']['uncertainty']['box_std_threshold'],
        entropy_threshold=config['training']['semi_supervised']['uncertainty']['entropy_threshold']
    )
    
    # MC Loss 함수
    mc_loss_fn = MCLoss(
        alpha=config['training']['mc_loss']['alpha'],
        beta=config['training']['mc_loss']['beta'],
        gamma=config['training']['mc_loss']['gamma'],
        temperature=config['training']['mc_loss']['temperature']
    )
    
    # 옵티마이저
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        momentum=config['training']['optimizer']['momentum'],
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    # 스케줄러
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    return {
        'model': model,
        'detector': detector,
        'mc_loss_fn': mc_loss_fn,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'labeled_dataset': labeled_dataset,
        'teacher_dataset': teacher_dataset,
        'val_transform': val_transform
    }


def train_worker(rank: int, world_size: int, gpu_ids: list, config: Dict[str, Any], args):
    """분산 학습 워커 함수"""
    
    try:
        # 분산 환경 설정
        if world_size > 1:
            device = setup_distributed_training(rank, world_size, gpu_ids, config)
        else:
            device = f'cuda:{gpu_ids[0]}' if gpu_ids else 'cpu'
            if device.startswith('cuda'):
                torch.cuda.set_device(int(device.split(':')[1]))
        
        # 로거 설정
        log_file = None
        if is_main_process():
            save_dir = Path(args.save_dir)
            run_dir = get_run_dir(args, config['model']['name'], config['data']['labeled_ratio'])
            save_dir = save_dir / run_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            log_file = str(save_dir / 'training.log')
        
        logger = setup_logger_distributed('train', log_file)
        
        # GPU 정보 로깅
        if is_main_process():
            log_gpu_info(logger)
            logger.info(f"Using device: {device}")
            logger.info(f"World size: {world_size}, Rank: {rank}")
        
        # 학습 객체들 생성
        training_objects = create_training_objects(config, device, rank)
        
        model = training_objects['model']
        detector = training_objects['detector']
        mc_loss_fn = training_objects['mc_loss_fn']
        optimizer = training_objects['optimizer']
        scheduler = training_objects['scheduler']
        labeled_dataset = training_objects['labeled_dataset']
        teacher_dataset = training_objects['teacher_dataset']
        
        # 모델을 분산 학습용으로 설정
        use_distributed = world_size > 1
        model = setup_model_for_distributed(
            model, 
            device, 
            use_distributed,
            find_unused_parameters=True
        )
        
        # 데이터 로더 생성
        memory_config = config['gpu']['memory']
        
        labeled_loader, unlabeled_loader, val_loader = labeled_dataset.get_dataloaders(
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers']
        )
        
        # 분산 학습용 데이터 로더로 변경
        if use_distributed:
            labeled_loader = setup_dataloader_for_distributed(
                labeled_dataset.labeled_dataset,
                batch_size=config['data']['batch_size'],
                num_workers=config['data']['num_workers'],
                use_distributed=True,
                shuffle=True,
                pin_memory=memory_config['pin_memory']
            )
            
            unlabeled_loader = setup_dataloader_for_distributed(
                labeled_dataset.unlabeled_dataset,
                batch_size=config['data']['batch_size'],
                num_workers=config['data']['num_workers'],
                use_distributed=True,
                shuffle=True,
                pin_memory=memory_config['pin_memory']
            )
            
            val_loader = setup_dataloader_for_distributed(
                labeled_dataset.val_dataset,
                batch_size=config['data']['batch_size'],
                num_workers=config['data']['num_workers'],
                use_distributed=True,
                shuffle=False,
                pin_memory=memory_config['pin_memory']
            )
        
        # Teacher용 unlabeled 데이터 로더 (pseudo label 생성용)
        teacher_unlabeled_loader = setup_dataloader_for_distributed(
            teacher_dataset.unlabeled_dataset,
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            use_distributed=use_distributed,
            shuffle=False,  # pseudo label 생성 시에는 순서 유지
            pin_memory=memory_config['pin_memory']
        ) if use_distributed else teacher_dataset.get_dataloaders(
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers']
        )[1]  # unlabeled_loader만 사용
        
        # Mixed Precision 스케일러 (AMP)
        scaler = torch.cuda.amp.GradScaler() if memory_config['mixed_precision'] and device.startswith('cuda') else None
        
        # 체크포인트 로드
        start_epoch = 0
        best_map = 0.0
        
        if args.resume and is_main_process():
            checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
            if checkpoint:
                start_epoch = checkpoint['epoch']
                best_map = checkpoint.get('best_map', 0.0)
                logger.info(f"Resumed from epoch {start_epoch}, best mAP: {best_map:.4f}")
        
        # 학습 루프
        logger.info("Starting training...")
        
        for epoch in range(start_epoch, config['training']['epochs']):
            
            # 분산 학습에서 sampler epoch 설정
            if use_distributed:
                if hasattr(labeled_loader.sampler, 'set_epoch'):
                    labeled_loader.sampler.set_epoch(epoch)
                if hasattr(unlabeled_loader.sampler, 'set_epoch'):
                    unlabeled_loader.sampler.set_epoch(epoch)
                if hasattr(teacher_unlabeled_loader.sampler, 'set_epoch'):
                    teacher_unlabeled_loader.sampler.set_epoch(epoch)
            
            # 학습 단계
            model.train()
            
            # Pseudo label 업데이트 (Teacher model 사용)
            if epoch >= config['training']['semi_supervised']['pseudo_label_start_epoch']:
                if is_main_process():
                    logger.info(f"Updating pseudo labels for epoch {epoch}")
                
                # Teacher 모델로 pseudo label 생성
                pseudo_labels = update_pseudo_labels(
                    model, 
                    teacher_unlabeled_loader,  # Teacher용 weak augmentation 데이터
                    detector, 
                    config['training']['semi_supervised']['conf_threshold'], 
                    device, 
                    config
                )
            
            # 한 에포크 학습
            train_losses = train_one_epoch(
                model=model,
                labeled_loader=labeled_loader,
                unlabeled_loader=unlabeled_loader,
                optimizer=optimizer,
                detector=detector,
                mc_loss_fn=mc_loss_fn,
                device=device,
                epoch=epoch,
                logger=logger,
                config=config,
                unlabeled_weight=config['training']['semi_supervised']['unlabeled_weight'],
                pseudo_label_start_epoch=config['training']['semi_supervised']['pseudo_label_start_epoch'],
                conf_threshold=config['training']['semi_supervised']['conf_threshold'],
                uncertainty_weight=config['training']['uncertainty_weight'],
                alignment_weight=config['model']['feature_alignment']['weight']
            )
            
            # 스케줄러 업데이트
            scheduler.step()
            
            # 검증 및 체크포인트 저장 (메인 프로세스만)
            if is_main_process() and (epoch + 1) % config['training']['val_interval'] == 0:
                # 검증 수행
                val_metrics = evaluate_model_with_memory_management(
                    model, val_loader, device, epoch, save_dir, args.val_data_path
                )
                
                current_map = val_metrics.get('mAP50', 0.0)
                is_best = current_map > best_map
                
                if is_best:
                    best_map = current_map
                
                # 체크포인트 저장
                if (epoch + 1) % config['training']['save_interval'] == 0:
                    checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
                    
                    # DDP 모델인 경우 module 속성 사용
                    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    
                    save_checkpoint_distributed({
                        'epoch': epoch + 1,
                        'model_state_dict': model_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_map': best_map,
                        'config': config
                    }, str(checkpoint_path), is_best)
                    
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                
                # MC Dropout 분석 저장
                save_mc_dropout_analysis(
                    model, detector, config, epoch, save_dir, logger, unlabeled_loader
                )
                
                logger.info(f"Epoch {epoch+1}/{config['training']['epochs']} - "
                          f"mAP50: {current_map:.4f}, Best: {best_map:.4f}")
        
        logger.info("Training completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise e
    
    finally:
        # 분산 환경 정리
        if world_size > 1:
            cleanup_distributed()


def main():
    """메인 함수"""
    args = parse_args()
    config = load_config(args.config, args)
    
    # GPU 설정 확인
    device, gpu_ids, use_distributed = setup_gpu_config(config)
    
    # 사용 가능한 포트 찾기
    if use_distributed and config['gpu']['distributed']['master_port'] == '12355':
        free_port = find_free_port()
        config['gpu']['distributed']['master_port'] = str(free_port)
        print(f"Using port: {free_port}")
    
    print(f"Available GPUs: {gpu_ids}")
    print(f"Using distributed training: {use_distributed}")
    
    # 분산 학습 시작
    if use_distributed:
        world_size = len(gpu_ids)
        print(f"Starting distributed training with {world_size} GPUs")
        
        # 멀티프로세싱 시작
        mp.spawn(
            train_worker,
            args=(world_size, gpu_ids, config, args),
            nprocs=world_size,
            join=True
        )
    else:
        # 단일 GPU 또는 CPU 학습
        print(f"Starting single process training on {device}")
        train_worker(0, 1, gpu_ids, config, args)


if __name__ == '__main__':
    # 멀티프로세싱을 위한 설정
    mp.set_start_method('spawn', force=True)
    main() 