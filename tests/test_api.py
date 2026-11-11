from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "MLOps API работает!"}


def test_predict_valid_data():
    # Имитируем правильный запрос пользователя
    payload = {
        "LIMIT_BAL": 50000.0,
        "AGE": 30,
        "EDUCATION": 2,
        "avg_payment": 1500.0,
        "age_group": "20-30"
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "default_prediction" in data
    assert "probability" in data


def test_predict_invalid_data():
    # Имитируем кривой запрос (отрицательный возраст)
    bad_payload = {
        "LIMIT_BAL": 50000.0,
        "AGE": -5,
        "EDUCATION": 2,
        "avg_payment": 1500.0,
        "age_group": "20-30"
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # Ошибка валидации