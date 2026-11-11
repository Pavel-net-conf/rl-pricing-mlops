# 🚀 Credit Default Prediction: MLOps Pipeline

Итоговый проект по дисциплине «Автоматизация процессов разработки и тестирования моделей машинного обучения». 
В проекте реализован полный цикл MLOps для задачи предсказания дефолта клиента (PD-модель) на базе открытого датасета UCI Credit Card.

## 🛠 Стек технологий
* **Машинное обучение:** Python, scikit-learn (GradientBoosting), Pandas, NumPy
* **Валидация данных:** Pandera
* **Трекинг экспериментов:** MLflow (SQLite backend)
* **Версионирование пайплайнов (CI):** DVC (Data Version Control)
* **Развертывание API (CD):** FastAPI, Uvicorn, Pydantic, Docker
* **Тестирование:** Pytest
* **Мониторинг:** Имитация и расчет дрейфа данных (Population Stability Index - PSI)

## 📁 Структура проекта
```text
rl-pricing-mlops/
├── data/
│   ├── raw/                # Исходный датасет (UCI_Credit_Card.csv)
│   └── processed/          # Обработанные данные (train.csv, test.csv)
├── models/                 # Обученные артефакты (credit_model.pkl)
├── scripts/                # Скрипты мониторинга (simulate_drift.py)
├── src/
│   ├── api/                # REST API сервис (app.py)
│   ├── data/               # Скрипты подготовки и валидации (make_dataset.py)
│   └── models/             # Пайплайн обучения модели (train.py)
├── tests/                  # Юнит-тесты API (test_api.py)
├── .venv/                  # Локальное виртуальное окружение
├── dvc.yaml                # Конфигурация DVC-пайплайна
├── Dockerfile              # Инструкции для сборки контейнера
├── requirements.txt        # Зависимости Python
└── README.md               # Документация проекта
'''


# 🚀 Инструкция по запуску
1. Подготовка окружения
Склонируйте репозиторий и настройте локальное виртуальное окружение (рекомендуется запускать на локальном диске, минуя облачные синхронизаторы).

Bash
# Создание и активация окружения (Windows)
python -m venv .venv
.venv\Scripts\activate

# Установка зависимостей
pip install --no-cache-dir -r requirements.txt
2. Воспроизведение пайплайна (DVC)
Проект использует DVC для связи этапов подготовки данных и обучения модели. Чтобы запустить весь конвейер одной командой:

Bash
dvc repro
Шаг prepare: Очистит данные, проведет feature engineering и проверит схему через Pandera.

Шаг train: Обучит модель GradientBoosting, подберет гиперпараметры и сохранит лучшую модель.

3. Трекинг экспериментов (MLflow)
Метрики, параметры и артефакты логируются локально. Для просмотра графического интерфейса:

Bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
Интерфейс будет доступен по адресу: http://127.0.0.1:5000

4. Запуск REST API (FastAPI)
Для локального запуска веб-сервиса с обученной моделью:

Bash
uvicorn src.api.app:app --reload
Документация и тестовый интерфейс (Swagger UI): http://127.0.0.1:8000/docs

5. Автоматическое тестирование
Проект покрыт юнит-тестами для проверки корректности работы API и валидации входных данных:

Bash
pytest tests/
6. Мониторинг дрейфа данных (Data Drift)
Скрипт мониторинга рассчитывает PSI (Population Stability Index) для выявления сдвига в распределении вероятностей дефолта при макроэкономических шоках:

Bash
python scripts/simulate_drift.py
7. Сборка Docker-контейнера
Для запуска приложения в полностью изолированной среде:

Bash
docker build -t credit_api .
docker run -p 8000:8000 credit_api
