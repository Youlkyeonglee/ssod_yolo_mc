from typing import Literal, Optional, Union, Dict, Any
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import yaml
from pathlib import Path

YOLOVersion = Literal["v8", "v9", "v10", "v11", "v12"]

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    YAML 설정 파일을 로드합니다.
    
    Args:
        config_path: YAML 설정 파일 경로
    
    Returns:
        설정 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_yolo_model(
    config_path: Optional[Union[str, Path]] = None,
    version: Optional[YOLOVersion] = "v8",
    model_size: Optional[Literal["n", "s", "m", "l", "x"]] = "m",
    task: Optional[Literal["detect", "segment", "pose", "classify"]] = "detect",
    pretrained: Optional[bool] = True
) -> YOLO:
    """
    YOLO 모델을 로드합니다.
    
    Args:
        config_path: YAML 설정 파일 경로 (설정 파일이 있는 경우 다른 인자들은 무시됨)
        version: YOLO 버전 (v8-v12)
        model_size: 모델 크기 (n: nano, s: small, m: medium, l: large, x: xlarge)
        task: 수행할 작업 (detect, segment, pose, classify)
        pretrained: 사전 학습된 가중치 사용 여부
    
    Returns:
        YOLO 모델 인스턴스
    """
    if config_path is not None:
        config = load_config(config_path)
        model_config = config['model']
        version = model_config['version']
        model_size = model_config['size']
        task = model_config['task']
        pretrained = model_config['pretrained']
    
    model_name = f"yolo{version}-{model_size}"
    if task != "detect":
        model_name = f"{model_name}-{task}"
    
    if pretrained:
        return YOLO(model_name)
    else:
        return YOLO(f"{model_name}.yaml")

def predict_image(
    model: YOLO,
    image: Union[str, np.ndarray],
    config_path: Optional[Union[str, Path]] = None,
    conf_threshold: Optional[float] = 0.25,
    device: Optional[str] = None
) -> list:
    """
    이미지에서 객체를 탐지합니다.
    
    Args:
        model: YOLO 모델 인스턴스
        image: 이미지 경로 또는 numpy 배열
        config_path: YAML 설정 파일 경로
        conf_threshold: 신뢰도 임계값
        device: 실행할 디바이스 (예: 'cpu', 'cuda:0')
    
    Returns:
        탐지 결과 리스트
    """
    if config_path is not None:
        config = load_config(config_path)
        inference_config = config['inference']
        conf_threshold = inference_config['conf_threshold']
        device = inference_config['device']
    
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    results = model.predict(
        source=image,
        conf=conf_threshold,
        device=device
    )
    return results

def process_video(
    model: YOLO,
    video_path: str,
    config_path: Optional[Union[str, Path]] = None,
    output_path: Optional[str] = None,
    conf_threshold: Optional[float] = 0.25,
    device: Optional[str] = None
) -> None:
    """
    비디오에서 객체를 탐지합니다.
    
    Args:
        model: YOLO 모델 인스턴스
        video_path: 비디오 파일 경로
        config_path: YAML 설정 파일 경로
        output_path: 결과 저장 경로
        conf_threshold: 신뢰도 임계값
        device: 실행할 디바이스
    """
    if config_path is not None:
        config = load_config(config_path)
        inference_config = config['inference']
        conf_threshold = inference_config['conf_threshold']
        device = inference_config['device']
        output_path = inference_config['save_dir']
    
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        device=device,
        save=True if output_path else False,
        project=output_path if output_path else None
    ) 