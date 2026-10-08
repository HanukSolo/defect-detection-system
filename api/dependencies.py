"""
Зависимости FastAPI: загрузка модели и трансформаций.
"""
import logging
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path

from src.config import (
    MODEL_SAVE_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES
)

logger = logging.getLogger(__name__)

_model = None
_transforms = None
_class_names = None


def get_transforms() -> transforms.Compose:
    global _transforms
    if _transforms is None:
        _transforms = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        logger.info("Трансформации для инференса загружены")
    return _transforms


def get_model(device: torch.device) -> nn.Module:
    global _model
    if _model is None:
        model_path = MODEL_SAVE_DIR / "best_resnet18_defect.pth"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Модель не найдена по пути: {model_path}. "
                "Сначала запустите `python train.py`"
            )

        logger.info(f"Загрузка модели из {model_path}...")

        # Создаем архитектуру (та же, что при обучении)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(in_features=512, out_features=NUM_CLASSES)

        # Загружаем веса с маппингом на устройство
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()  # ВАЖНО: режим инференса!

        _model = model
        logger.info(f"Модель загружена на устройство: {device}")

    return _model


def get_class_names() -> list:
    global _class_names
    if _class_names is None:
        _class_names = [
            'crazing', 'inclusion', 'patches',
            'pitted_surface', 'rolled-in_scale', 'scratches'
        ]
    return _class_names