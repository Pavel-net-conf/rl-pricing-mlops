import pandas as pd
import os
from sklearn.model_selection import train_test_split


def prepare_data():
    raw_path = "data/raw/UCI_Credit_Card.csv"
    if not os.path.exists(raw_path):
        print(f"Положите файл с Kaggle по пути: {raw_path}")
        return

    print("Загрузка данных...")
    data = pd.read_csv(raw_path)

    if 'default.payment.next.month' in data.columns:
        data = data.rename(columns={'default.payment.next.month': 'target'})
    if 'ID' in data.columns:
        data = data.drop('ID', axis=1)

    print("Feature Engineering...")
    # биннинг возраста
    data['age_group'] = pd.cut(
        data['AGE'],
        bins=[20, 30, 40, 50, 100],
        labels=['20-30', '30-40', '40-50', '50+']
    )

    # агрегация платежей за 6 месяцев
    pay_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    data['avg_payment'] = data[pay_cols].mean(axis=1)

    print("Разбиение на train/test (80/20)...")
    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print(f"Готово! Train: {train_df.shape}, Test: {test_df.shape}")


if __name__ == '__main__':
    prepare_data()
