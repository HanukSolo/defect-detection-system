"""
Модуль для работы с данными: Dataset, DataLoader, трансформации.
"""
import logging
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.config import (
    RAW_DATA_DIR, IMAGE_SIZE, BATCH_SIZE, TRAIN_VAL_SPLIT,
    NUM_WORKERS, IMAGENET_MEAN, IMAGENET_STD
)

logger = logging.getLogger(__name__)


def get_transforms(is_training: bool = True) -> transforms.Compose:
    if is_training:
        # Для обучения: аугментации + нормализация
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        # Для валидации/теста: только ресайз и нормализация
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def get_data_loaders() -> tuple[DataLoader, DataLoader, list]:
    logger.info(f"Загрузка данных из {RAW_DATA_DIR}")

    full_dataset = datasets.ImageFolder(
        root=RAW_DATA_DIR,
        transform=get_transforms(is_training=True)
    )

    class_names = full_dataset.classes
    logger.info(f"Найдены классы: {class_names}")

    train_size = int(TRAIN_VAL_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.transform = get_transforms(is_training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    logger.info(f"Размер train набора: {len(train_dataset)} изображений")
    logger.info(f"Размер val набора: {len(val_dataset)} изображений")

    return train_loader, val_loader, class_names