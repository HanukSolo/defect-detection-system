"""
Pydantic схемы для API ответов.
"""
from pydantic import BaseModel, Field
from typing import List


class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., description="Название предсказанного класса дефекта")
    confidence: float = Field(..., description="Уверенность модели (вероятность) от 0 до 1")
    all_probabilities: dict = Field(..., description="Вероятности по всем классам")
    message: str = Field(..., description="Человекочитаемое сообщение")


class ErrorResponse(BaseModel):
    error: str
    detail: str