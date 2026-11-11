# Используем легковесный официальный образ Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код API и сохраненную модель
COPY src/api/ ./src/api/
COPY models/credit_model.pkl ./models/

# Открываем порт 8000, на котором работает FastAPI
EXPOSE 8000

# Запускаем uvicorn-сервер
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]