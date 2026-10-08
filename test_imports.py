"""
Проверка, что все библиотеки установлены корректно.
"""
import torch
import torchvision
import fastapi
import pydantic
import streamlit
import pytest
import matplotlib
import seaborn
import sklearn
import PIL
import cv2
import prometheus_client
import onnx
import onnxruntime
import requests

print("✅ Все библиотеки установлены успешно!")
print(f"PyTorch версия: {torch.__version__}")
print(f"torchvision версия: {torchvision.__version__}")
print(f"CUDA доступен: {torch.cuda.is_available()}")