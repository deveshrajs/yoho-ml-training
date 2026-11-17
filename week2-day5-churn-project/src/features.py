"""
Telecommunications Customer Churn Feature Engineering Module.
Implements a scikit-learn compatible transformer for production pipelines.
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TelecomFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer that computes domain-specific behavioral
    and financial metrics for telecom subscribers without data leakage.
    """

    def __init__(self, include_interaction_terms: bool = True):
        self.include_interaction_terms = include_interaction_terms
        self.service_columns = [
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies',
        ]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit transformer (stateless)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input DataFrame by engineering lifecycle, spend velocity,
        and ecosystem bundle metrics.
        """
        df = X.copy()

        # 1. Clean numeric TotalCharges (blank spaces from tenure=0 become 0.0)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)

        # 2. Tenure Lifecycle Cohorts
        if 'tenure' in df.columns:
            df['tenure_cohort'] = pd.cut(
                df['tenure'],
                bins=[-1, 12, 24, 48, 72],
                labels=['0_12m_Onboarding', '13_24m_Established', '25_48m_Mature', '49_72m_Loyal']
            ).astype(str)

            # 3. Monthly Spend Velocity & Rate Drift
            # If customer has tenure > 0, compute historical effective monthly spend
            effective_monthly_spend = np.where(
                df['tenure'] > 0,
                df['TotalCharges'] / df['tenure'],
                df.get('MonthlyCharges', 0.0)
            )
            df['effective_monthly_spend'] = effective_monthly_spend

            if 'MonthlyCharges' in df.columns:
                # Difference between current monthly charge and historical average spend
                df['monthly_rate_delta'] = df['MonthlyCharges'] - df['effective_monthly_spend']

        # 4. Digital Ecosystem & Service Depth
        existing_services = [col for col in self.service_columns if col in df.columns]
        if existing_services:
            # Count subscribed add-on services ('Yes' = 1, otherwise 0)
            df['addon_services_count'] = (df[existing_services] == 'Yes').sum(axis=1)
            
            # Security & Support bundle indicator
            sec_cols = [c for c in ['OnlineSecurity', 'TechSupport'] if c in df.columns]
            if sec_cols:
                df['has_support_security'] = ((df[sec_cols] == 'Yes').sum(axis=1) >= 1).astype(int)

        return df
