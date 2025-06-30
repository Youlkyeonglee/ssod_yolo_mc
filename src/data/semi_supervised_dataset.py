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
        self.val_root = Path(self.config['data']['val']['images'])
        self.class_names = self.config['data']['names']
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
        
        print(f"Loading labeled data from: {labeled_list}")
        print(f"Loading unlabeled data from: {unlabeled_list}")
        
        # 실제 COCO 데이터 경로
        coco_data_root = "/media/lee/Data/COCO/train2017"
        
        # 먼저 레이블 데이터 존재 여부 확인
        print("\n=== 레이블 데이터 검증 ===")
        self._check_label_availability(coco_data_root)
        
        # 데이터셋 초기화 - 실제 이미지/레이블 경로로 수정
        print("\n=== Labeled 데이터셋 초기화 ===")
        self.labeled_dataset = YOLODataset(
            data_root=coco_data_root,
            image_list=self._get_image_list(labeled_list),
            has_labels=True,
            transform=self.transform,
            max_samples=self.max_samples,
        )
        
        print(f"✓ Labeled 데이터셋: {len(self.labeled_dataset)} 샘플")
        
        print("\n=== Unlabeled 데이터셋 초기화 ===")
        self.unlabeled_dataset = YOLODataset(
            data_root=coco_data_root,
            image_list=self._get_image_list(unlabeled_list),
            has_labels=False,  # unlabeled 데이터는 레이블 검증 안함
            transform=self.transform,
            max_samples=self.max_samples,
            class_names=self.class_names
        )
        
        print(f"✓ Unlabeled 데이터셋: {len(self.unlabeled_dataset)} 샘플")
        
        # 검증 데이터셋 초기화
        print("\n=== Validation 데이터셋 초기화 ===")
        self.val_dataset = YOLODataset(
            data_root=self.val_root,
            image_list=None,  # 전체 디렉토리 사용
            has_labels=True,
            transform=self.transform,
            max_samples=self.max_samples,
            class_names=self.class_names    
        )
        
        print(f"✓ Validation 데이터셋: {len(self.val_dataset)} 샘플")
        
        # 데이터셋 유효성 최종 확인
        if len(self.labeled_dataset) == 0:
            raise ValueError("❌ Labeled 데이터셋이 비어있습니다! 레이블 파일을 확인하세요.")
        
        if len(self.unlabeled_dataset) == 0:
            raise ValueError("❌ Unlabeled 데이터셋이 비어있습니다! 데이터 리스트 파일을 확인하세요.")
    
    def _check_label_availability(self, data_root: str):
        """레이블 파일 존재 여부와 구조 확인"""
        from pathlib import Path
        
        data_path = Path(data_root)
        images_dir = data_path / "images"
        labels_dir = data_path / "labels"
        
        print(f"데이터 루트: {data_path}")
        print(f"이미지 디렉토리: {images_dir} (존재: {images_dir.exists()})")
        print(f"레이블 디렉토리: {labels_dir} (존재: {labels_dir.exists()})")
        
        if images_dir.exists():
            image_count = len(list(images_dir.glob("*.jpg")))
            print(f"이미지 파일 수: {image_count}")
        
        if labels_dir.exists():
            label_count = len(list(labels_dir.glob("*.txt")))
            print(f"레이블 파일 수: {label_count}")
            
            # 샘플 레이블 파일 검증
            sample_labels = list(labels_dir.glob("*.txt"))[:3]
            for label_file in sample_labels:
                try:
                    if label_file.stat().st_size > 0:
                        labels = np.loadtxt(str(label_file))
                        if labels.ndim == 1:
                            labels = labels.reshape(1, -1)
                        print(f"✓ {label_file.name}: {labels.shape[0]} 객체, {labels.shape[1]} 차원")
                    else:
                        print(f"✓ {label_file.name}: 빈 파일 (배경 이미지)")
                except Exception as e:
                    print(f"✗ {label_file.name}: 오류 - {e}")
        else:
            print("⚠️  레이블 디렉토리가 존재하지 않습니다!")
            
        print("=" * 50)
    
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
        max_samples: Optional[int] = None,
        class_names: Optional[List[str]] = None
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
        
        # COCO 클래스 이름 설정
        self.class_names = class_names
        
        # 이미지 리스트 설정
        if image_list is None:
            # 전체 디렉토리 스캔
            self.image_paths = list(self.data_root.glob('*.jpg'))
        else:
            self.image_paths = [Path(p) for p in image_list]
        
        # print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        # print(self.data_root)
        # print(len(self.image_paths))
        # 레이블이 있는 데이터셋인 경우 유효한 레이블을 가진 이미지만 필터링
        if self.has_labels:
            # 레이블 파일 리스트 생성 (COCO 구조에 맞게 수정)
            label_files = []
            for img_file in self.image_paths:
                # 절대 경로를 사용하여 레이블 파일 경로 생성
                img_path_str = str(img_file)
                if "/images/" in img_path_str:
                    label_file = img_path_str.replace("/images/", "/labels/").replace(".jpg", ".txt")
                else:
                    # 백업 로직: 단순 교체
                    label_file = img_path_str.replace(".jpg", ".txt").replace("images", "labels")
                label_files.append(label_file)
            
            print(f"Sample image path: {self.image_paths[0] if self.image_paths else 'None'}")
            print(f"Sample label path: {label_files[0] if label_files else 'None'}")
            
            # 유효한 레이블을 가진 이미지만 필터링
            valid_image_paths = []
            valid_label_files = []
            
            for img_path, label_file in zip(self.image_paths, label_files):
                if self._has_valid_labels(label_file):
                    valid_image_paths.append(img_path)
                    valid_label_files.append(label_file)
            
            self.image_paths = valid_image_paths
            self.label_files = valid_label_files
            
            print(f"Filtered dataset: {len(self.image_paths)} images with valid labels out of {len(label_files)} total images")
            
            # 디버깅: 필터링 과정 상세 로그
            if len(self.image_paths) < len(label_files):
                filtered_out = len(label_files) - len(self.image_paths)
                print(f"⚠️  {filtered_out}개 이미지가 유효하지 않은 레이블로 인해 필터링되었습니다.")
                
                # 필터링된 이미지들의 이유 분석 (처음 5개만)
                filtered_samples = []
                for img_path, label_file in zip(self.image_paths[:5] if len(self.image_paths) >= 5 else [], label_files[:5]):
                    if not self._has_valid_labels(label_file):
                        filtered_samples.append((img_path, label_file))
                
                if filtered_samples:
                    print("필터링된 샘플들:")
                    for img_path, label_file in filtered_samples:
                        print(f"  - {Path(img_path).name} -> {Path(label_file).name}")
            
            # 레이블 파일이 하나도 없으면 에러 처리
            if len(self.image_paths) == 0:
                print("❌ ERROR: No valid labels found!")
                print(f"Expected label directory: {Path(label_files[0]).parent if label_files else 'Unknown'}")
                
                # 레이블 디렉토리 존재 여부 확인
                if label_files:
                    expected_label_dir = Path(label_files[0]).parent
                    if not expected_label_dir.exists():
                        print(f"❌ 레이블 디렉토리가 존재하지 않습니다: {expected_label_dir}")
                        print("💡 COCO 데이터셋 구조를 확인하세요:")
                        print("   - images/ 디렉토리에 .jpg 파일들")
                        print("   - labels/ 디렉토리에 .txt 파일들")
                        raise FileNotFoundError(f"레이블 디렉토리를 찾을 수 없습니다: {expected_label_dir}")
                    else:
                        print(f"⚠️  레이블 디렉토리는 존재하지만 유효한 레이블 파일이 없습니다: {expected_label_dir}")
                        # 레이블 파일 수 확인
                        label_file_count = len(list(expected_label_dir.glob("*.txt")))
                        print(f"디렉토리 내 .txt 파일 수: {label_file_count}")
                        
                        if label_file_count == 0:
                            raise FileNotFoundError(f"레이블 디렉토리에 .txt 파일이 없습니다: {expected_label_dir}")
                        else:
                            print("⚠️  레이블 파일들이 유효하지 않은 형식입니다. 첫 몇 개 확인 중...")
                            sample_labels = list(expected_label_dir.glob("*.txt"))[:5]
                            for label_file in sample_labels:
                                print(f"  - {label_file.name}: 크기 {label_file.stat().st_size} bytes")
                            raise ValueError("유효한 레이블 파일을 찾을 수 없습니다. YOLO 형식 확인 필요.")
                
                raise ValueError("레이블된 데이터셋을 초기화할 수 없습니다.")
        
        # 테스트 모드: 최대 샘플 수 제한
        if max_samples is not None and max_samples > 0:
            self.image_paths = self.image_paths[:max_samples]
            if self.has_labels:
                self.label_files = self.label_files[:max_samples]
    
    def _has_valid_labels(self, label_file: str) -> bool:
        """레이블 파일이 유효한 데이터를 가지고 있는지 확인"""
        label_path = Path(label_file)
        
        # 파일이 존재하지 않으면 유효하지 않음
        if not label_path.exists():
            print(f"Label file not found: {label_path}")
            return False
            
        try:
            # 파일이 비어있는지 확인 - 빈 파일도 허용
            if label_path.stat().st_size == 0:
                return True  # 빈 레이블 파일도 유효한 것으로 처리
                
            # 실제로 로드해보고 유효한 데이터인지 확인
            labels = np.loadtxt(str(label_path))
            
            # 빈 배열이면 유효하지 않음 - 하지만 허용
            if labels.size == 0:
                return True  # 빈 레이블도 유효한 것으로 처리
                
            # 1차원 배열인 경우 2차원으로 변환
            if len(labels.shape) == 1:
                labels = labels.reshape(1, -1)
                
            # YOLO 형식 검증: 각 행이 5개 값을 가져야 함 (class, x, y, w, h)
            if labels.shape[1] != 5:
                print(f"Invalid label format in {label_path}: expected 5 columns, got {labels.shape[1]}")
                return False
                
            # 클래스 ID 검증 (모든 값 허용, 학습 시 클램핑으로 처리)
            class_ids = labels[:, 0]
            # 음수 클래스 ID만 실제로 문제가 되므로 이것만 체크
            if np.any(class_ids < 0):
                print(f"Invalid class IDs in {label_path}: negative class IDs found")
                return False
                
            # 좌표값 검증: 박스가 이미지 경계를 심하게 벗어나는지 확인
            for i, label in enumerate(labels):
                x_center, y_center, width, height = label[1:5]
                
                # 박스 경계 계산 (center + width/height 형식)
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                # 심각한 경계 위반 체크를 더 관대하게 수정
                if (x1 < -1.0 or x2 > 2.0 or y1 < -1.0 or y2 > 2.0):
                    print(f"Invalid bbox in {label_path} object {i}: bbox severely out of bounds")
                    print(f"  x_center={x_center:.3f}, y_center={y_center:.3f}, w={width:.3f}, h={height:.3f}")
                    print(f"  calculated bounds: x1={x1:.3f}, y1={y1:.3f}, x2={x2:.3f}, y2={y2:.3f}")
                    return False
                
                # width/height가 너무 큰 경우 체크를 더 관대하게
                if width > 3.0 or height > 3.0:
                    print(f"Invalid bbox size in {label_path} object {i}: width={width:.3f}, height={height:.3f}")
                    return False
                
                # 음수 크기 체크
                if width <= 0 or height <= 0:
                    print(f"Invalid bbox size in {label_path} object {i}: width={width:.3f}, height={height:.3f} (non-positive)")
                    return False
                
            return True
            
        except (ValueError, OSError, np.DataError) as e:
            # 파일 읽기 오류나 형식 오류가 있으면 유효하지 않음
            print(f"Error reading label file {label_path}: {e}")
            return False
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        # 이미지 로드
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # numpy array를 PIL Image로 변환 (torchvision transform 호환을 위해)
        img = Image.fromarray(img)
        
        # 레이블 로드 (있는 경우)
        labels = None
        if self.has_labels and hasattr(self, 'label_files'):
            # 이미 필터링된 유효한 레이블만 남아있으므로 안전하게 로드
            label_file_path = self.label_files[idx]
            if Path(label_file_path).stat().st_size > 0:  # 빈 파일이 아닌 경우
                labels = np.loadtxt(str(label_file_path))
                # 단일 객체인 경우 2D 배열로 변환
                if len(labels.shape) == 1:
                    labels = labels.reshape(1, -1)
                
                # 레이블 후처리: 클램핑 제거 - 원본 레이블 그대로 사용
                # (시각화에서만 경계 처리)
        
        # PyTorch transform 적용 (이미지만)
        if self.transform:
            img = self.transform(img)
        
        # 레이블이 없으면 빈 배열 생성
        if labels is None:
            labels = np.zeros((0, 5))
        
        return {
            'image': img,
            'labels': torch.from_numpy(labels.astype(np.float32)),
            'img_path': str(img_path)
        }
    
    def _clamp_labels(self, labels: np.ndarray) -> np.ndarray:
        """레이블 좌표를 유효한 범위로 클램핑 (center + width/height 형식)"""
        clamped_labels = labels.copy()
        
        for i in range(len(clamped_labels)):
            class_id = clamped_labels[i, 0]
            x_center, y_center, width, height = clamped_labels[i, 1:5]
            
            # 박스 크기 클램핑 (최대 1.0으로 제한)
            width = min(width, 1.0)
            height = min(height, 1.0)
            
            # center 좌표 클램핑 (박스가 이미지 경계를 벗어나지 않도록)
            x_center = max(width / 2, min(1.0 - width / 2, x_center))
            y_center = max(height / 2, min(1.0 - height / 2, y_center))
            
            # 최종 박스 경계 계산 및 검증
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # 경계 체크 및 추가 클램핑
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(1.0, x2)
            y2 = min(1.0, y2)
            
            # 클램핑된 박스에서 center와 크기 재계산
            width = x2 - x1
            height = y2 - y1
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            
            # 최소 박스 크기 보장 (너무 작으면 제거)
            if width < 0.01 or height < 0.01:
                # 유효하지 않은 박스 제거를 위해 클래스 ID를 -1로 설정
                class_id = -1
            
            # 클램핑된 값 할당
            clamped_labels[i, 0] = class_id
            clamped_labels[i, 1] = x_center
            clamped_labels[i, 2] = y_center
            clamped_labels[i, 3] = width
            clamped_labels[i, 4] = height
        
        # 유효하지 않은 박스들 (class_id == -1) 제거
        valid_mask = clamped_labels[:, 0] >= 0
        clamped_labels = clamped_labels[valid_mask]
        
        return clamped_labels
    
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