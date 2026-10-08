"""
Модуль для оценки и визуализации результатов обучения.
"""
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_training_history(history: dict, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # График Loss
    axes[0].plot(history['train_losses'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_losses'], label='Val Loss', linewidth=2)
    axes[0].set_title('Loss over Epochs', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # График Accuracy
    axes[1].plot(history['train_accs'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_accs'], label='Val Acc', linewidth=2)
    axes[1].set_title('Accuracy over Epochs', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"График сохранен в {save_path}")

    plt.show()