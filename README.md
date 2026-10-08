# 🛡️ Defect Detection System

Production-ready ML-сервис для классификации дефектов металлических поверхностей с использованием Transfer Learning (ResNet18), развернутый в Kubernetes с полным циклом мониторинга (Prometheus + Grafana).

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Minikube-326ce5.svg)](https://kubernetes.io/)

---

## 📋 Описание

Система автоматически анализирует изображения металлических поверхностей и классифицирует их по 6 типам дефектов: *crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches*. 

Проект демонстрирует полный жизненный цикл ML-модели: от EDA и обучения с Transfer Learning до контейнеризации, оркестрации в Kubernetes и настройки observability.

### 🌟 Ключевые особенности
- **Высокая точность**: 99.4% Validation Accuracy благодаря Transfer Learning (ResNet18) и аугментациям.
- **Модульная архитектура**: Чистый код, разделенный на `src/` (config, dataset, model, train) по принципам SOLID.
- **Production-ready API**: FastAPI с валидацией входных файлов, асинхронной обработкой и Swagger UI.
- **Оркестрация**: Деплой в Kubernetes с 2 репликами, Liveness/Readiness пробами и ограничением ресурсов.
- **Мониторинг**: Интеграция с Prometheus (кастомные метрики RPS, Latency, Error Rate) и визуализация в Grafana.

---

## 🏗️ Архитектура системы

### Схема взаимодействия компонентов
```
👤 Клиент (Swagger UI)
│
▼ HTTP POST /predict
🔄 K8s Service (LoadBalancer)
│
├──▶ Pod 1: FastAPI + ResNet18 ──┐
│ │
└──▶ Pod 2: FastAPI + ResNet18 ──┤
│
▼
📊 Prometheus
(scrape /metrics)
│
▼
📈 Grafana Dashboard
```
### Компоненты системы

| Компонент | Технология | Назначение |
| --- | --- | --- |
| **API-сервер** | FastAPI + Uvicorn | REST API для инференса модели |
| **ML-модель** | PyTorch ResNet18 | Классификация 6 типов дефектов |
| **Оркестратор** | Kubernetes (Minikube) | Управление подами, масштабирование, health-checks |
| **Контейнеризация** | Docker | Изоляция зависимостей и воспроизводимость |
| **Мониторинг** | Prometheus + Grafana | Сбор метрик и визуализация |
| **ServiceMonitor** | Custom Resource | Автоматическое обнаружение сервиса для Prometheus |

---

## 🛠️ Технологический стек

| Категория | Технологии |
|-----------|------------|
| **ML / CV** | PyTorch, Torchvision, ResNet18, Pillow, Albumentations (опц.) |
| **Backend** | FastAPI, Uvicorn, Pydantic, Python-Multipart |
| **Инфраструктура** | Docker, Kubernetes (Minikube), Helm |
| **Мониторинг** | Prometheus, Grafana, Prometheus-Client |
| **Инструменты** | Git, PyCharm, Jupyter Notebook |

---

## 📂 Структура проекта

```text
defect-detection-system/
├── api/                    # FastAPI приложение
│   ├── main.py             # Точки входа и эндпоинты
│   ├── dependencies.py     # Загрузка модели и трансформаций (singleton)
│   └── schemas.py          # Pydantic модели ответов
├── src/                    # Модули машинного обучения
│   ├── config.py           # Централизованная конфигурация
│   ├── dataset.py          # DataLoader и аугментации
│   ├── model.py            # Архитектура ResNet18 с Transfer Learning
│   ├── train.py            # Цикл обучения
│   └── evaluate.py         # Визуализация метрик обучения
├── k8s/                    # Kubernetes манифесты
│   ├── deployment.yaml     # Деплой с 2 репликами и probes
│   ├── service.yaml        # Внутренний балансировщик
│   └── servicemonitor.yaml # Конфигурация сбора метрик Prometheus
├── data/                   # Данные и артефакты
│   ├── raw/                # Реорганизованный датасет NEU-DET
│   └── models/             # Сохраненные веса (.pth)
├── Dockerfile              # Инструкции для сборки образа
├── requirements.txt        # Зависимости Python
└── README.md               # Этот файл
```


##  Мониторинг и Метрики

Система экспортирует кастомные метрики для оценки здоровья ML-сервиса в реальном времени через эндпоинт `/metrics`:

| Метрика | Тип | Описание |
|---------|-----|----------|
| `http_requests_total` | Counter | Общее количество запросов (с разбивкой по методу, эндпоинту и статус-коду) |
| `http_request_duration_seconds` | Histogram | Гистограмма задержки инференса (P50, P90, P99) |

### Сбор метрик

- **ServiceMonitor** автоматически обнаруживает сервис через label `app: defect-detection`
- Prometheus скрейпит метрики каждые **15 секунд**
- Grafana визуализирует данные в реальном времени

### Ключевые дашборды

#### Requests Per Second (RPS)

```promql
sum(rate(http_requests_total{exported_endpoint="/predict"}[5m]))
```

#### Average Prediction Latency
```promql
sum(rate(http_request_duration_seconds_sum{exported_endpoint="/predict"}[5m])) 
/ 
sum(rate(http_request_duration_seconds_count{exported_endpoint="/predict"}[5m]))
```

#### Error Rate (%)
```promql
sum(rate(http_requests_total{http_status=~"4..|5..", exported_endpoint="/predict"}[5m])) 
/ 
sum(rate(http_requests_total{exported_endpoint="/predict"}[5m])) 
* 100
```

#### Total Requests
```promql
sum(http_requests_total{exported_endpoint="/predict"})
```

### Примеры метрик в действии
- **Средняя задержка инференса**: ~91 мс на запрос (CPU)
- **Обработано запросов в тесте**: 8 запросов без ошибок
- **Error Rate**: 0% (все запросы вернули статус 200

## 📈 Результаты модели
- **Архитектура**: ResNet18 (предобучена на ImageNet, заморожены базовые слои)
- **Dataset**: NEU-DET (6 классов, ~1800 изображений, балансировка 80/20)
- **Валидационная точность**: 99.4%
- **Средняя задержка инференса (CPU)**: ~91 мс на запрос
- **Масштабирование**: 2 реплики в Kubernetes для отказоустойчивости

## 👤 Автор
**Арсений Гирин**
[ML Engineer]
https://github.com/HanukSolo
