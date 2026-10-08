"""
Скрипт для реорганизации датасета NEU-DET в структуру ImageFolder.
Из: data/raw/NEU-DET/{train,validation}/images/{class}/{num}.jpg
В:  data/raw/NEU-DET-organized/{class}/{class}_{num}.jpg
"""
import os
import shutil
from pathlib import Path

# Пути (используем абсолютный путь от корня проекта)
PROJECT_ROOT = Path(__file__).parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "NEU-DET"
ORGANIZED_DIR = PROJECT_ROOT / "data" / "raw" / "NEU-DET-organized"

# Ожидаемые классы
EXPECTED_CLASSES = [
    'crazing', 'inclusion', 'patches',
    'pitted_surface', 'rolled-in_scale', 'scratches'
]

def reorganize():
    """Перекладывает картинки по папкам классов."""
    # Создаём целевую структуру
    for cls in EXPECTED_CLASSES:
        (ORGANIZED_DIR / cls).mkdir(parents=True, exist_ok=True)

    # Обрабатываем train и validation
    for split in ['train', 'validation']:
        img_dir = RAW_DIR / split / 'images'
        if not img_dir.exists():
            print(f"⚠️ Папка не найдена: {img_dir}")
            continue

        print(f"\n📂 Обрабатываем {split}/images...")

        # Проходим по папкам классов внутри images/
        for cls in EXPECTED_CLASSES:
            cls_dir = img_dir / cls
            if not cls_dir.exists():
                print(f"   ⚠️ Папка класса не найдена: {cls_dir}")
                continue

            files = list(cls_dir.iterdir())
            print(f"   {cls}: найдено {len(files)} файлов")

            for img_path in files:
                # Проверяем, что это файл (а не папка)
                if img_path.is_file():
                    # Копируем файл в целевую папку
                    dest = ORGANIZED_DIR / cls / img_path.name
                    shutil.copy2(img_path, dest)

    # Считаем итоговую статистику
    print("\n✅ Реорганизация завершена! Статистика:")
    total = 0
    for cls in EXPECTED_CLASSES:
        cls_dir = ORGANIZED_DIR / cls
        count = len([f for f in cls_dir.iterdir() if f.is_file()])
        total += count
        print(f"   {cls}: {count} изображений")
    print(f"\n Всего изображений: {total}")

if __name__ == "__main__":
    reorganize()