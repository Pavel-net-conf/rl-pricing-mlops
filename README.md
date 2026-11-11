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
