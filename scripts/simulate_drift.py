import numpy as np
import pandas as pd
import joblib


def calculate_psi(expected, actual, buckets=10):
    """Расчет PSI (Population Stability Index) для вероятностей дефолта."""
    # Разбиваем ожидаемые вероятности 
    expected_pct = np.histogram(expected, bins=buckets, range=(0, 1))[0] / len(expected)
    # Разбиваем новые вероятности 
    actual_pct = np.histogram(actual, bins=buckets, range=(0, 1))[0] / len(actual)

    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi_value


def run_simulation():
    print("Загрузка модели и данных...")
    model = joblib.load("models/credit_model.pkl")

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    # эталонные вероятности 
    train_probs = model.predict_proba(train_df.drop("target", axis=1))[:, 1]

    #  имитация входящего потока данных 
    print("Имитация нормального трафика (Тестовая выборка)...")
    test_probs = model.predict_proba(test_df.drop("target", axis=1))[:, 1]
    normal_psi = calculate_psi(train_probs, test_probs)
    print(f"PSI нормального трафика: {normal_psi:.4f} (Менее 0.1 - дрифта нет)")

    # имитация шоковый сценарий
    print("\nИмитация шокового трафика (Доходы снижены на 50%)...")
    anomalous_df = test_df.copy()
    anomalous_df["avg_payment"] = anomalous_df["avg_payment"] * 0.5

    anomalous_probs = model.predict_proba(anomalous_df.drop("target", axis=1))[:, 1]
    drift_psi = calculate_psi(train_probs, anomalous_probs)

    print(f"PSI шокового трафика: {drift_psi:.4f}")
    if drift_psi > 0.2:
        print("ALARM: Обнаружен значительный дрифт данных! Модель нужно переобучить.")
    elif drift_psi > 0.1:
        print("WARNING: Умеренный дрифт данных.")
    else:
        print("OK: Распределения стабильны.")


if __name__ == "__main__":
    run_simulation()
