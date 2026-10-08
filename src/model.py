"""
Модуль для создания модели с Transfer Learning.
"""
import logging
import torch
import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES

logger = logging.getLogger(__name__)


def create_model(device: torch.device, freeze_base: bool = True) -> nn.Module:
    logger.info("Загрузка предобученной модели ResNet18...")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_base:
        logger.info("Замораживаем базовые слои (обучаем только classifier head)")
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(in_features=num_ftrs, out_features=NUM_CLASSES)

    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)

    logger.info(f"Модель создана: {NUM_CLASSES} классов, устройство: {device}")
    logger.info(f"Последний слой: {model.fc}")

    return model