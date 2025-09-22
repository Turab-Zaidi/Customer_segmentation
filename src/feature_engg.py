import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

class FeatureEngineer:
    def __init__(self, scaler_path="models/scaler.pkl",pca_path="models/pca.pkl"):
        self.scaler_path = scaler_path
        self.pca_path = pca_path
        self._pca = PCA(n_components=0.95)
        self._scaler = StandardScaler() 
        self.right_skewed_cols = ['BALANCE','ONEOFF_PURCHASES_FREQUENCY','PURCHASES_TRX']
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)

    def _create_and_select_features(self, df: pd.DataFrame) -> pd.DataFrame:


        df['ONEOFF_PURCHASE_RATIO'] = df['ONEOFF_PURCHASES'] / (df['PURCHASES'] + 1e-6)
        df['INSTALLMENT_PURCHASE_RATIO'] = df['INSTALLMENTS_PURCHASES'] / (df['PURCHASES'] + 1e-6)
        df['CASH_ADVANCE_TRX_RATIO'] = df['CASH_ADVANCE_TRX'] / (df['PURCHASES_TRX'] + df['CASH_ADVANCE_TRX'] + 1e-6)
        df['PAYMENT_TO_MIN_PAYMENT_RATIO'] = df['PAYMENTS'] / (df['MINIMUM_PAYMENTS'] + 1e-6)
        df['CREDIT_UTILIZATION'] = df['BALANCE'] / (df['CREDIT_LIMIT'] + 1e-6)
        df['AVG_PURCHASE_VALUE'] = df['PURCHASES'] / (df['PURCHASES_TRX'] + 1e-6)
        
        drop_cols = [
            'CUST_ID',                              # identifier
            'ONEOFF_PURCHASES',                     # captured by ONEOFF_PURCHASE_RATIO
            'INSTALLMENTS_PURCHASES',               # captured by INSTALLMENT_PURCHASE_RATIO
            'PURCHASES_INSTALLMENTS_FREQUENCY',     # highly correlated with PURCHASES_FREQUENCY
            'CASH_ADVANCE_TRX',                      # highly correlated with CASH_ADVANCE_FREQUENCY
            'BALANCE_FREQUENCY',                     # low impact
            'CREDIT_LIMIT',                          # captured by CREDIT_UTILIZATION
            'PAYMENTS'                               # captured by PAYMENT_RATIO
        ]       
        df = df.drop(columns=drop_cols, errors='ignore')
        
        for col in self.right_skewed_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col])
        
        return df

    def fit(self, df: pd.DataFrame):
        """Fits the entire pipeline: processes data, fits scaler, and saves it."""
        df_processed = self._create_and_select_features(df.copy())
        
        numeric_cols_ = df_processed.select_dtypes(include=np.number).columns
        
        self._scaler.fit(df_processed[numeric_cols_])
        joblib.dump(self._scaler, self.scaler_path)

        df_scaled = self._scaler.transform(df_processed[numeric_cols_])
        
        self._pca.fit(df_scaled)
        joblib.dump(self._pca, self.pca_path)
        

    def _load_model(self):
        """Loads the fitted scaler from the path."""
        try:
            return joblib.load(self.scaler_path), joblib.load(self.pca_path)
        except FileNotFoundError:
            raise RuntimeError(f"MOdel file not found . Please run fit() first.")

    def transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the pipeline using the SAVED scaler. This is for inference.
        """
        scaler,pca = self._load_model()
        
        df_processed = self._create_and_select_features(df.copy())
        

        scaled_data = scaler.transform(df_processed)
        pca_data = pca.transform(scaled_data)

        pca_cols = [f'PC_{i+1}' for i in range(pca.n_components_)]
        df_final = pd.DataFrame(pca_data, index=df.index, columns=pca_cols)
        
        return df_final
                

