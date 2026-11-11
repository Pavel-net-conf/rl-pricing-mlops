import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    RocCurveDisplay
)
import joblib
import os


def train_model():
    print("Загружаем обработанные данные...")
    train_df = pd.read_csv("data/processed/train.csv")

    # 1. Подготовка признаков и таргета
    X = train_df.drop("target", axis=1)
    y = train_df["target"]

    # Откусим еще 20% от train_df для честной отложенной проверки (test_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Собираем Sklearn Pipeline...")
    # Указываем, какие колонки числовые, а какие категориальные
    numeric_features = ["LIMIT_BAL", "AGE", "avg_payment"]
    categorical_features = ["EDUCATION", "age_group"]

    # Обработка чисел: заполняем пропуски медианой и масштабируем
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Обработка категорий: заполняем пропуски частым значением и делаем OneHot
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Объединяем оба трансформера
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Финальный Pipeline: предобработка + классификатор
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=42))
    ])

    print("Настраиваем поиск гиперпараметров (GridSearchCV)...")
    # Задаем сетку для поиска лучших параметров
    param_grid = {
        "classifier__n_estimators": [50, 100],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [3, 5]
    }

    # GridSearchCV переберет все комбинации (5+ экспериментов для MLflow)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1
    )

    print("Запускаем MLflow и обучение...")
    # Настраиваем локальную папку для логов MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Credit_Default_Optimization")

    with mlflow.start_run(run_name="GridSearch_Run"):
        # Обучаем сразу всё: от заполнения пропусков до перебора параметров
        grid_search.fit(X_train, y_train)

        # Достаем лучшую модель
        best_model = grid_search.best_estimator_
        print(f"Лучшие параметры: {grid_search.best_params_}")

        print("Считаем метрики на отложенной выборке...")
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred))
        }

        print(f"Результаты: {metrics}")

        # Логируем лучшие параметры и метрики в MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)

        print("Строим ROC-кривую и сохраняем артефакты...")
        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=ax)
        plt.title("ROC Curve - Credit Default Model")

        # Сохраняем картинку локально и отправляем в MLflow
        roc_path = "roc_curve.png"
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)
        plt.close()

        # Сохраняем саму модель в MLflow, разрешая загрузку деревьев scikit-learn
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="best_credit_model",
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"]
        )

        # Также сохраняем .pkl файл в папку models (он пригодится для API)
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/credit_model.pkl")

        print("Готово! Эксперимент залогирован, модель сохранена.")


if __name__ == "__main__":
    train_model()

