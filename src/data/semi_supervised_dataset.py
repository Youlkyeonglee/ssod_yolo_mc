from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import cv2
import yaml
from PIL import Image

class SemiSupervisedDataset:
    """Semi-supervised 학습을 위한 데이터셋 관리자"""
    
    def __init__(
        self,
        config_path: Union[str, Path],
        percent: float = 1.0,
        seed: int = 1,
        transform=None,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            config_path: 설정 파일 경로
            percent: 레이블된 데이터 비율 (1.0, 2.0, 5.0, 10.0)
            seed: 데이터 분할 시드
            transform: 데이터 변환 함수
        """
        self.config = self._load_config(config_path)
        self.transform = transform
        self.percent = percent
        self.seed = seed
        self.max_samples = max_samples
        # 데이터 경로 설정
        self.data_root = Path(self.config['data']['root'])
        self.val_root = Path(self.config['data']['val']['root'])
        
        # 데이터셋 초기화
        self.labeled_dataset = None
        self.unlabeled_dataset = None
        self.val_dataset = None
        
        self._initialize_datasets()
    
    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """설정 파일 로드"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_image_list(self, list_file: Path) -> List[str]:
        """텍스트 파일에서 이미지 리스트 로드"""
        if not list_file.exists():
            raise FileNotFoundError(f"데이터 리스트 파일을 찾을 수 없습니다: {list_file}")
        
        with open(list_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def _initialize_datasets(self):
        """데이터셋 초기화"""
        # 레이블된/레이블되지 않은 데이터 리스트 파일
        labeled_list = self.data_root / f"COCO_train2017_p{self.percent}_s{self.seed}_labeled_data.txt"
        unlabeled_list = self.data_root / f"COCO_train2017_p{self.percent}_s{self.seed}_unlabeled_data.txt"
        
        # 데이터셋 초기화
        self.labeled_dataset = YOLODataset(
            data_root=self.data_root,
            image_list=self._get_image_list(labeled_list),
            has_labels=True,
            transform=self.transform,
            max_samples=self.max_samples
        )
        
        self.unlabeled_dataset = YOLODataset(
            data_root=self.data_root,
            image_list=self._get_image_list(unlabeled_list),
            has_labels=False,
            transform=self.transform,
            max_samples=self.max_samples
        )
        
        # 검증 데이터셋 초기화
        self.val_dataset = YOLODataset(
            data_root=self.val_root,
            image_list=None,  # 전체 디렉토리 사용
            has_labels=True,
            transform=self.transform,
            max_samples=self.max_samples
        )
    
    def get_dataloaders(
        self,
        batch_size: int = 16,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """데이터 로더 생성
        
        Returns:
            labeled_loader, unlabeled_loader, val_loader
        """
        labeled_loader = DataLoader(
            self.labeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.labeled_dataset.collate_fn
        )
        
        unlabeled_loader = DataLoader(
            self.unlabeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.unlabeled_dataset.collate_fn
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.val_dataset.collate_fn
        )
        
        return labeled_loader, unlabeled_loader, val_loader

class YOLODataset(Dataset):
    """YOLO 형식의 데이터셋"""
    
    def __init__(
        self,
        data_root: Union[str, Path],
        image_list: Optional[List[str]] = None,
        has_labels: bool = True,
        transform=None,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_root: 데이터 루트 디렉토리
            image_list: 이미지 파일 경로 리스트
            has_labels: 레이블 파일 존재 여부
            transform: 데이터 변환 함수
            max_samples: 최대 샘플 수 (테스트용)
        """
        self.data_root = Path(data_root)
        self.has_labels = has_labels
        self.transform = transform
        
        # 이미지 리스트 설정
        if image_list is None:
            # 전체 디렉토리 스캔
            self.image_paths = list(self.data_root.glob('*.jpg'))
        else:
            self.image_paths = [Path(p) for p in image_list]
        
        # 테스트 모드: 최대 샘플 수 제한
        if max_samples is not None and max_samples > 0:
            self.image_paths = self.image_paths[:max_samples]
        
        if self.has_labels:
            # 레이블 파일 리스트 생성
            self.label_files = [
                self.data_root / "labels" / f"{img_file.stem}.txt"
                for img_file in self.image_paths
            ]
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        # 이미지 로드
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 레이블 로드 (있는 경우)
        labels = None
        if self.has_labels and self.label_files[idx].exists():
            labels = np.loadtxt(str(self.label_files[idx]))
            # 단일 객체인 경우 2D 배열로 변환
            if len(labels.shape) == 1:
                labels = labels.reshape(1, -1)
        
        # 데이터 변환 적용
        if self.transform:
            # YOLO 형식: [class_id, x_center, y_center, width, height]
            transformed = self.transform(
                image=img,
                bboxes=[] if labels is None else labels[:, 1:].tolist(),  # bbox 좌표만
                class_labels=[] if labels is None else labels[:, 0].astype(int).tolist()  # 클래스 ID
            )
            img = transformed['image']
            
            # 변환된 바운딩 박스와 클래스 레이블 결합
            if labels is not None and len(transformed['bboxes']) > 0:
                bboxes = np.array(transformed['bboxes'])  # [x_center, y_center, width, height]
                class_ids = np.array(transformed['class_labels']).reshape(-1, 1)  # [class_id]
                labels = np.hstack([class_ids, bboxes])  # [class_id, x_center, y_center, width, height]
            else:
                labels = np.zeros((0, 5))  # 빈 레이블의 경우 5열로 설정
        
        return {
            'image': img,
            'labels': torch.from_numpy(labels if labels is not None else np.zeros((0, 5))),  # 항상 torch.Tensor로 반환
            'img_path': str(img_path)
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """배치 데이터 처리"""
        images = torch.stack([item['image'] for item in batch])
        labels = [item['labels'] for item in batch]  # 이미 torch.Tensor
        img_paths = [item['img_path'] for item in batch]
        
        return {
            'images': images,
            'labels': labels,
            'img_paths': img_paths
        } 