"""
FastAPI приложение для сервиса детекции дефектов.
"""
import logging
import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from PIL import Image
import torch
import torch.nn.functional as F
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from api.dependencies import get_model, get_transforms, get_class_names
from api.schemas import PredictionResponse, ErrorResponse
from src.config import CLASS_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP Requests',
    ['method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP Request Latency',
    ['endpoint']
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Инициализация сервиса детекции дефектов...")
    get_model(DEVICE)
    get_transforms()
    get_class_names()
    logger.info("Сервис готов к работе!")
    yield
    logger.info("Завершение работы сервиса...")

app = FastAPI(
    title="Defect Detection API",
    description="ML сервис для детекции дефектов на металлических поверхностях",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": True
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}
)
async def predict_defect(file: UploadFile = File(...)):
    start_time = time.time()
    status_code = 200

    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_extension = Path(file.filename).suffix.lower() if file.filename else ""

    if file_extension not in allowed_extensions:
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', http_status='400').inc()
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат: {file_extension}. Допустимые: {allowed_extensions}"
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        transform = get_transforms()
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        model = get_model(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
        class_names = get_class_names()
        predicted_class = class_names[predicted_idx]

        all_probabilities = {
            class_names[i]: round(prob.item(), 4)
            for i, prob in enumerate(probabilities)
        }

        if confidence >= 0.9:
            message = f"Высокая уверенность: обнаружен дефект '{predicted_class}'"
        elif confidence >= 0.7:
            message = f"Средняя уверенность: вероятен дефект '{predicted_class}'"
        else:
            message = f"Низкая уверенность: возможный дефект '{predicted_class}', требуется проверка"

        REQUEST_COUNT.labels(method='POST', endpoint='/predict', http_status='200').inc()
        REQUEST_LATENCY.labels(endpoint='/predict').observe(time.time() - start_time)

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            all_probabilities=all_probabilities,
            message=message
        )

    except Exception as e:
        logger.error(f"Ошибка при предсказании: {str(e)}")
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', http_status='500').inc()
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {str(e)}")