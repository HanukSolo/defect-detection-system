"""
Конфигурация проекта Defect Detection System.
Централизованное хранилище всех параметров и путей.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "NEU-DET-organized"
MODEL_SAVE_DIR = DATA_DIR / "models"

IMAGE_SIZE = 224  # ResNet ожидает 224x224
BATCH_SIZE = 32
TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% val
NUM_WORKERS = 0  # Для Windows лучше 0, для Linux можно 2-4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 6  # crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches

CLASS_NAMES = [
    'crazing',
    'inclusion',
    'patches',
    'pitted_surface',
    'rolled-in_scale',
    'scratches'
]