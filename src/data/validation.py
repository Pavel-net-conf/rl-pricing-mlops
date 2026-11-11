import pandera.pandas as pa
from pandera import Column, Check
import pandas as pd

# Схема ожидаемых данных
pd_schema = pa.DataFrameSchema({
    "LIMIT_BAL": Column(float, Check.ge(0.0), nullable=False),
    "AGE": Column(int, Check.in_range(20, 100), nullable=False),
    "EDUCATION": Column(int, Check.in_range(0, 6), nullable=False),
    "target": Column(int, Check.isin([0, 1]), required=False)
})

def validate_data(df):
    return pd_schema.validate(df)

if __name__ == '__main__':
    print("Валидация train.csv...")
    try:
        train_df = pd.read_csv("data/processed/train.csv")
        validate_data(train_df)
        print("Успех: данные прошли проверку.")
    except Exception as e:
        print("Ошибка валидации!")
        print(e)