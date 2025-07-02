import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Optional, Tuple, Dict, Any
import logging
import socket
from pathlib import Path


def is_gpu_available() -> bool:
    """GPU 사용 가능 여부 확인"""
    return torch.cuda.is_available()


def get_available_gpus() -> List[int]:
    """사용 가능한 GPU ID 리스트 반환"""
    if not is_gpu_available():
        return []
    return list(range(torch.cuda.device_count()))


def setup_gpu_config(config: Dict[str, Any]) -> Tuple[str, List[int], bool]:
    """
    GPU 설정 구성
    
    Args:
        config: 설정 딕셔너리
    
    Returns:
        device: 주요 디바이스 ('cpu' 또는 'cuda:0')
        gpu_ids: 사용할 GPU ID 리스트
        use_distributed: 분산 학습 사용 여부
    """
    gpu_config = config.get('gpu', {})
    
    # GPU 사용 여부 확인
    use_gpu = gpu_config.get('use_gpu', True) and is_gpu_available()
    
    if not use_gpu:
        return 'cpu', [], False
    
    # GPU ID 설정
    if gpu_config.get('auto_detect', True):
        gpu_ids = get_available_gpus()
    else:
        specified_ids = gpu_config.get('gpu_ids', [0])
        available_ids = get_available_gpus()
        gpu_ids = [gpu_id for gpu_id in specified_ids if gpu_id in available_ids]
    
    if not gpu_ids:
        return 'cpu', [], False
    
    # 주요 디바이스 설정
    device = f'cuda:{gpu_ids[0]}'
    
    # 분산 학습 여부 결정
    distributed_config = gpu_config.get('distributed', {})
    use_distributed = (
        distributed_config.get('enabled', True) and 
        len(gpu_ids) > 1
    )
    
    return device, gpu_ids, use_distributed


def setup_distributed_training(
    rank: int,
    world_size: int,
    gpu_ids: List[int],
    config: Dict[str, Any]
) -> str:
    """
    분산 학습 환경 설정
    
    Args:
        rank: 현재 프로세스 순위
        world_size: 전체 프로세스 수
        gpu_ids: 사용할 GPU ID 리스트
        config: 설정 딕셔너리
    
    Returns:
        device: 현재 프로세스가 사용할 디바이스
    """
    distributed_config = config['gpu']['distributed']
    
    # 환경 변수 설정
    os.environ['MASTER_ADDR'] = distributed_config.get('master_addr', 'localhost')
    os.environ['MASTER_PORT'] = str(distributed_config.get('master_port', '12355'))
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    
    # 분산 프로세스 그룹 초기화
    backend = distributed_config.get('backend', 'nccl')
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    # 현재 프로세스가 사용할 GPU 설정
    local_gpu_id = gpu_ids[rank % len(gpu_ids)]
    device = f'cuda:{local_gpu_id}'
    torch.cuda.set_device(local_gpu_id)
    
    return device


def cleanup_distributed():
    """분산 학습 환경 정리"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """메인 프로세스 여부 확인"""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size() -> int:
    """전체 프로세스 수 반환"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """현재 프로세스 순위 반환"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def setup_model_for_distributed(
    model: torch.nn.Module,
    device: str,
    use_distributed: bool,
    find_unused_parameters: bool = True,
    force_single_gpu: bool = False  # 단일 GPU 강제 사용 플래그 추가
) -> torch.nn.Module:
    """
    모델을 분산 학습용으로 설정
    
    Args:
        model: 학습할 모델
        device: 디바이스
        use_distributed: 분산 학습 사용 여부
        find_unused_parameters: 사용되지 않는 파라미터 찾기 여부
        force_single_gpu: 단일 GPU 강제 사용 (MC Dropout 호환성을 위해)
    
    Returns:
        설정된 모델
    """
    model = model.to(device)
    
    if use_distributed and dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(device.split(':')[1])],
            find_unused_parameters=find_unused_parameters
        )
    elif torch.cuda.device_count() > 1 and not use_distributed and not force_single_gpu:
        # DataParallel 사용 (분산 학습이 아니고 단일 GPU 강제가 아닌 경우)
        print("⚠️  Multi-GPU detected but not using DataParallel for MC Dropout compatibility")
        # model = torch.nn.DataParallel(model)  # MC Dropout 호환성을 위해 비활성화
    
    return model


def setup_dataloader_for_distributed(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    use_distributed: bool,
    shuffle: bool = True,
    pin_memory: bool = True,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    분산 학습용 DataLoader 설정
    
    Args:
        dataset: 데이터셋
        batch_size: 배치 크기
        num_workers: 워커 프로세스 수
        use_distributed: 분산 학습 사용 여부
        shuffle: 데이터 셔플 여부
        pin_memory: 메모리 고정 여부
    
    Returns:
        설정된 DataLoader
    """
    sampler = None
    if use_distributed and dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=shuffle
        )
        shuffle = False  # sampler를 사용할 때는 shuffle=False
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        **kwargs
    )


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    분산 환경에서 텐서를 모든 프로세스에서 집계
    
    Args:
        tensor: 집계할 텐서
        average: 평균을 구할지 여부
    
    Returns:
        집계된 텐서
    """
    if not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    
    if average:
        rt /= get_world_size()
    
    return rt


def save_checkpoint_distributed(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False
):
    """
    분산 환경에서 체크포인트 저장 (메인 프로세스만)
    
    Args:
        state: 저장할 상태
        filepath: 저장 경로
        is_best: 최고 성능 모델 여부
    """
    if is_main_process():
        torch.save(state, filepath)
        if is_best:
            best_filepath = filepath.replace('.pth', '_best.pth')
            torch.save(state, best_filepath)


def print_distributed(message: str):
    """분산 환경에서 메인 프로세스만 출력"""
    if is_main_process():
        print(message)


def setup_logger_distributed(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """분산 환경용 로거 설정"""
    logger = logging.getLogger(name)
    
    if is_main_process():
        logger.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
    else:
        logger.setLevel(logging.WARNING)
    
    return logger


def check_port_availability(port: int, host: str = 'localhost') -> bool:
    """포트 사용 가능 여부 확인"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


def find_free_port(start_port: int = 12355, max_attempts: int = 100) -> int:
    """사용 가능한 포트 찾기"""
    for port in range(start_port, start_port + max_attempts):
        if check_port_availability(port):
            return port
    raise RuntimeError(f"Could not find a free port in range {start_port}-{start_port + max_attempts}")


def get_gpu_memory_info() -> Dict[int, Dict[str, float]]:
    """GPU 메모리 정보 조회"""
    gpu_info = {}
    
    if not is_gpu_available():
        return gpu_info
    
    for gpu_id in range(torch.cuda.device_count()):
        torch.cuda.set_device(gpu_id)
        
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3  # GB
        allocated_memory = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
        cached_memory = torch.cuda.memory_reserved(gpu_id) / 1024**3  # GB
        
        gpu_info[gpu_id] = {
            'total': total_memory,
            'allocated': allocated_memory,
            'cached': cached_memory,
            'free': total_memory - cached_memory
        }
    
    return gpu_info


def log_gpu_info(logger: logging.Logger):
    """GPU 정보 로깅"""
    if not torch.cuda.is_available():
        logger.info("GPU를 사용할 수 없습니다.")
        return
    
    logger.info(f"사용 가능한 GPU 수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")


def distributed_main(parse_args_func, main_func):
    """분산 학습을 위한 메인 함수"""
    import torch.multiprocessing as mp
    
    # 명령행 인자와 설정 파싱
    args, config = parse_args_func()
    
    # GPU 설정 확인
    device, gpu_ids, use_distributed = setup_gpu_config(config)
    
    print(f"사용 가능한 GPU: {gpu_ids}")
    print(f"분산 학습 사용: {use_distributed}")
    
    if use_distributed and len(gpu_ids) > 1:
        # 포트 찾기
        try:
            free_port = find_free_port()
            config['gpu']['distributed']['master_port'] = str(free_port)
            print(f"사용할 포트: {free_port}")
        except Exception as port_error:
            print(f"포트 찾기 실패: {port_error}")
            print("기본 포트 12355 사용")
            config['gpu']['distributed']['master_port'] = '12355'
        
        # 멀티프로세싱으로 분산 학습 시작
        world_size = len(gpu_ids)
        print(f"분산 학습 시작: {world_size}개 프로세스")
        
        try:
            mp.spawn(
                distributed_worker,
                args=(world_size, gpu_ids, config, args, main_func),
                nprocs=world_size,
                join=True
            )
            print("분산 학습 완료")
            return  # 성공적으로 완료된 경우 여기서 종료
        except Exception as e:
            print(f"❌ 분산 학습 spawn 오류: {e}")
            print("단일 GPU로 대체 실행")
            import traceback
            traceback.print_exc()
            # 분산 학습 실패 시 첫 번째 GPU로 단일 학습
            if gpu_ids:
                args.device = f"cuda:{gpu_ids[0]}"
                print(f"단일 GPU 학습으로 전환: {args.device}")
                main_func()
            else:
                print("사용 가능한 GPU가 없어 CPU로 실행")
                args.device = "cpu"
                main_func()
    else:
        # 단일 GPU 또는 CPU 학습
        print("단일 프로세스 학습 시작")
        if gpu_ids:
            args.device = f"cuda:{gpu_ids[0]}"
        else:
            args.device = "cpu"
        
        # main_func이 인자를 받는지 확인
        if hasattr(main_func, '__name__') and main_func.__name__ == 'main_distributed':
            # main_distributed 함수인 경우 인자를 전달
            main_func(args, config, use_distributed=False)
        else:
            # 일반 main 함수인 경우
            main_func()


def distributed_worker(rank: int, world_size: int, gpu_ids: list, config: dict, args, main_distributed_func):
    """분산 학습 워커 함수"""
    try:
        print(f"🚀 워커 {rank}/{world_size} 시작 (GPU: {gpu_ids})")
        
        # 분산 환경 설정
        device = setup_distributed_training(rank, world_size, gpu_ids, config)
        print(f"🔧 워커 {rank}: 디바이스 {device} 설정 완료")
        
        # args 디바이스 업데이트
        args.device = device
        
        # 메인 학습 로직 실행
        print(f"📚 워커 {rank}: 학습 시작")
        
        # main_distributed_func가 팩토리 함수에서 생성된 함수인지 확인
        if hasattr(main_distributed_func, '__name__') and main_distributed_func.__name__ == 'main_distributed':
            # 팩토리에서 생성된 main_distributed 함수인 경우
            main_distributed_func(args, config, use_distributed=True)
        else:
            # 일반 main 함수인 경우 (use_distributed 인자 없이 호출)
            main_distributed_func()
        
        print(f"✅ 워커 {rank}: 학습 완료")
        
    except KeyboardInterrupt:
        print(f"⚠️  워커 {rank}: 사용자 중단")
    except Exception as e:
        print(f"❌ 워커 {rank} 오류: {e}")
        import traceback
        print(f"워커 {rank} 상세 오류:")
        traceback.print_exc()
        
        # Inplace operation 오류인 경우 특별 처리
        if "inplace operation" in str(e):
            print(f"🚨 워커 {rank}: Inplace operation 오류 감지 - 안전한 종료 시도")
            try:
                # GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cleanup_distributed()
            except Exception as cleanup_error:
                print(f"⚠️  워커 {rank} 정리 중 추가 오류: {cleanup_error}")
        
        raise e
    finally:
        print(f"🧹 워커 {rank}: 정리 중...")
        try:
            cleanup_distributed()
        except Exception as cleanup_error:
            print(f"⚠️  워커 {rank} 정리 오류: {cleanup_error}")
        print(f"👋 워커 {rank}: 종료")


def create_main_distributed_wrapper(
    get_run_dir_func,
    setup_logger_func,
    YOLOWithMCDropout_class,
    MCDropoutDetector_class,
    SemiSupervisedDataset_class,
    train_one_epoch_func,
    evaluate_model_wrapper_func,
    get_val_transform_func
):
    """main_distributed 함수를 생성하는 팩토리 함수"""
    
    def main_distributed(args, config, use_distributed=False):
        """분산 학습용 메인 함수"""
        import torch.optim as optim
        
        # 실행 디렉토리 설정 (메인 프로세스만)
        if is_main_process():
            run_dir = get_run_dir_func(args, args.model, args.labeled_ratio)
            logger = setup_logger_func(run_dir)
            logger.info(f"Results will be saved to: {run_dir}")
            log_gpu_info(logger)
        else:
            # 워커 프로세스는 간단한 로거만
            import logging
            logger = logging.getLogger(f'worker_{torch.distributed.get_rank()}')
            logger.setLevel(logging.WARNING)
            run_dir = Path(".")  # 임시
        
        # MC Dropout이 적용된 YOLO 모델 생성
        ema_decay = config.get('model', {}).get('ema', {}).get('decay', 0.999)
        model = YOLOWithMCDropout_class(
            model_name=args.model,
            dropout_rate=args.dropout_rate,
            feature_alignment_enabled=args.feature_alignment_enabled,
            num_classes=config['data']['nc'],
            ema_decay=ema_decay
        )
        
        # 모델을 분산 학습용으로 설정 (MC Dropout 호환성 고려)
        model = setup_model_for_distributed(
            model, 
            args.device, 
            use_distributed,
            find_unused_parameters=True,
            force_single_gpu=not use_distributed  # 단일 GPU 시 DataParallel 비활성화
        )
        
        if is_main_process():
            logger.info(f"🔄 EMA Teacher 업데이트 활성화: decay={ema_decay}")
        
        # MC Dropout 탐지기 생성
        detector = MCDropoutDetector_class(
            model=model,
            num_samples=args.num_samples,
            dropout_rate=args.dropout_rate,
            box_std_threshold=args.box_std_threshold,
            entropy_threshold=args.entropy_threshold,
            save_dir=run_dir / "mcdropout" if is_main_process() else None
        )
        
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
        
        # 데이터셋 초기화
        try:
            base_transform = get_val_transform_func(img_size=config['data']['img_size'])
            
            dataset_manager = SemiSupervisedDataset_class(
                config_path=args.config,
                percent=args.labeled_ratio,
                seed=args.seed,
                transform=base_transform,
                max_samples=args.max_samples
            )
            
            # 분산 학습용 데이터 로더 생성
            if use_distributed:
                memory_config = config.get('gpu', {}).get('memory', {})
                pin_memory = memory_config.get('pin_memory', True)
                
                labeled_loader = setup_dataloader_for_distributed(
                    dataset_manager.labeled_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    use_distributed=True,
                    shuffle=True,
                    pin_memory=pin_memory,
                    collate_fn=dataset_manager.labeled_dataset.collate_fn
                )
                
                unlabeled_loader = setup_dataloader_for_distributed(
                    dataset_manager.unlabeled_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    use_distributed=True,
                    shuffle=True,
                    pin_memory=pin_memory,
                    collate_fn=dataset_manager.unlabeled_dataset.collate_fn
                )
                
                val_loader = setup_dataloader_for_distributed(
                    dataset_manager.val_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    use_distributed=True,
                    shuffle=False,
                    pin_memory=pin_memory,
                    collate_fn=dataset_manager.val_dataset.collate_fn
                )
            else:
                labeled_loader, unlabeled_loader, val_loader = dataset_manager.get_dataloaders(
                    batch_size=args.batch_size,
                    num_workers=args.num_workers
                )
            
            if is_main_process():
                logger.info(f"✓ Labeled 데이터: {len(labeled_loader.dataset)} 샘플")
                logger.info(f"✓ Unlabeled 데이터: {len(unlabeled_loader.dataset)} 샘플")
                logger.info(f"✓ Validation 데이터: {len(val_loader.dataset)} 샘플")
            
            # 분산 학습에서도 GT 데이터 시각화 수행 (메인 프로세스만) - 임시 비활성화
            if is_main_process():
                logger.info("=== GT 데이터 시각화 건너뜀 (분산 학습) ===")
                logger.warning("GT 데이터 시각화가 hang 현상을 일으켜서 일시적으로 비활성화되었습니다.")
                logger.info("학습이 완료된 후 별도로 시각화를 실행할 수 있습니다.")
        
        except Exception as e:
            if is_main_process():
                logger.error(f"❌ 데이터셋 초기화 실패: {e}")
            raise
        
        # 학습 루프
        if is_main_process():
            logger.info("Starting distributed training...")
        
        for epoch in range(args.epochs):
            
            # 분산 학습에서 sampler epoch 설정
            if use_distributed:
                if hasattr(labeled_loader.sampler, 'set_epoch'):
                    labeled_loader.sampler.set_epoch(epoch)
                if hasattr(unlabeled_loader.sampler, 'set_epoch'):
                    unlabeled_loader.sampler.set_epoch(epoch)
            
            # 학습 단계
            avg_loss = train_one_epoch_func(
                model=model,
                labeled_loader=labeled_loader,
                unlabeled_loader=unlabeled_loader,
                optimizer=optimizer,
                detector=detector,
                device=args.device,
                epoch=epoch,
                logger=logger,
                config=config,
                unlabeled_weight=args.unlabeled_weight,
                pseudo_label_start_epoch=args.pseudo_label_start_epoch,
                conf_threshold=args.conf_threshold,
                alignment_weight=args.feature_alignment_weight if args.feature_alignment_enabled else 0.0
            )
            
            if is_main_process():
                logger.info(f"📊 Epoch {epoch+1} - Average Loss: {avg_loss:.6f}")
            
            # 스케줄러 업데이트
            scheduler.step()
            
            # 검증 및 저장 (메인 프로세스만)
            if is_main_process() and (epoch + 1) % config['training']['val_interval'] == 0:
                try:
                    mAP50, mAP50_95 = evaluate_model_wrapper_func(
                        model=model,
                        val_loader=val_loader,
                        device=args.device,
                        epoch=epoch,
                        save_dir=run_dir,
                        val_data_path=args.val_data_path
                    )
                    
                    logger.info(f"📊 Epoch {epoch+1} - mAP@0.5: {mAP50:.4f}")
                    
                    # 체크포인트 저장
                    if (epoch + 1) % config['training']['save_interval'] == 0:
                        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                        
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': model_state,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': avg_loss,
                            'mAP50': mAP50,
                            'config': config
                        }
                        
                        checkpoint_path = run_dir / f'checkpoint_epoch_{epoch+1}.pth'
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
                        
                except Exception as e:
                    logger.error(f"검증 실패: {e}")
    
    return main_distributed 