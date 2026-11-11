#  легковесный официальный образ Python
FROM python:3.10-slim

# установка рабочей директории внутри контейнера
WORKDIR /app

#  файл с зависимостями и установка
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#  исходный код API и сохраненная модель
COPY src/api/ ./src/api/
COPY models/credit_model.pkl ./models/

# открываем порт 8000, на котором работает FastAPI
EXPOSE 8000

# запуск uvicorn-сервер
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
