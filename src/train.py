"""
Модуль для обучения модели.
"""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_DIR

logger = logging.getLogger(__name__)


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        num_epochs: int = NUM_EPOCHS,
        learning_rate: float = LEARNING_RATE
) -> dict:
    criterion = nn.CrossEntropyLoss()

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=learning_rate)

    best_val_acc = 0.0
    best_model_wts = None

    history = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': []
    }

    logger.info(f"Начинаем обучение на {num_epochs} эпох...")
    logger.info(f"Количество обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logger.info("-" * 10)

        # --- TRAIN PHASE ---
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = correct_preds.double() / total_samples

        # --- VALIDATION PHASE ---
        model.eval()
        val_running_loss = 0.0
        val_correct_preds = 0
        val_total_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct_preds += torch.sum(preds == labels.data)
                val_total_samples += inputs.size(0)

        epoch_val_loss = val_running_loss / val_total_samples
        epoch_val_acc = val_correct_preds.double() / val_total_samples

        history['train_losses'].append(epoch_train_loss)
        history['val_losses'].append(epoch_val_loss)
        history['train_accs'].append(epoch_train_acc.item())
        history['val_accs'].append(epoch_val_acc.item())

        logger.info(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        logger.info(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info(f"✨ Новая лучшая модель! Val Acc: {best_val_acc:.4f}")

    model.load_state_dict(best_model_wts)

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_SAVE_DIR / "best_resnet18_defect.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"\n🏆 Обучение завершено! Лучшая модель сохранена в {model_path}")
    logger.info(f"Лучшая Val Accuracy: {best_val_acc:.4f}")

    return history