import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self):
        pass

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates()

        numeric_cols = df.select_dtypes(include='number').columns
        categorical_cols = df.select_dtypes(include='object').columns

        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

        return df


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.clean_data(df)
        return df
