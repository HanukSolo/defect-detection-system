"""
Главный скрипт для обучения модели детекции дефектов.
"""
import logging
import torch

from src.config import MODEL_SAVE_DIR
from src.dataset import get_data_loaders
from src.model import create_model
from src.train import train_model
from src.evaluate import plot_training_history

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 50)
    logger.info("ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ ДЕТЕКЦИИ ДЕФЕКТОВ")
    logger.info("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используем устройство: {device}")

    train_loader, val_loader, class_names = get_data_loaders()

    model = create_model(device, freeze_base=True)

    history = train_model(model, train_loader, val_loader, device)

    plot_training_history(history, save_path=MODEL_SAVE_DIR / "training_history.png")

    logger.info("=" * 50)
    logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()