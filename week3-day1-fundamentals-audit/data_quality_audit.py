"""
Data Quality Audit, Sanitization, and Learning Curve Analysis Module.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_data_audit():
    cities = ['Chennai', 'chennai', 'CHENNAI ', 'Madurai', 'madurai ', 'Coimbatore', ' Coimbatore']
    np.random.seed(42)

    base_df = pd.DataFrame({
        'customer_id': list(range(1001, 1030)),
        'full_name': [f'Enterprise_User_{i}' for i in range(1, 30)],
        'email_address': [f'client_{i}@enterprise.in' if i % 5 != 0 else np.nan for i in range(1, 30)],
        'city_region': [cities[i % len(cities)] for i in range(29)],
        'customer_age': ([27, 33, -4, 48, 31, 245, 29, 42] * 4)[:29],
        'contact_phone': ['9876543210' if i % 6 != 0 else '4321' for i in range(1, 30)]
    })

    # Add duplicate rows
    df_raw = pd.concat([base_df, base_df.iloc[[2, 7, 14]]], ignore_index=True)
    df_raw.to_csv('customers_raw.csv', index=False)

    total_rows = len(df_raw)
    duplicates = df_raw.duplicated().sum()
    missing_email_pct = (df_raw['email_address'].isna().sum() / total_rows) * 100
    invalid_ages = ((df_raw['customer_age'] < 18) | (df_raw['customer_age'] > 100)).sum()
    invalid_phones = (df_raw['contact_phone'].astype(str).str.strip().str.len() != 10).sum()

    print("=== RAW DATA QUALITY AUDIT ===")
    print(f"Ingested Records      : {total_rows}")
    print(f"Exact Duplicate Rows  : {duplicates}")
    print(f"Missing Email Address : {missing_email_pct:.1f}%")
    print(f"Outlier Invalid Ages  : {invalid_ages}")
    print(f"Malformed Phones      : {invalid_phones}")

    # Cleaning operations
    df_clean = df_raw.drop_duplicates().copy().reset_index(drop=True)
    df_clean['city_region'] = df_clean['city_region'].astype(str).str.strip().str.title()
    df_clean['email_missing_flag'] = df_clean['email_address'].isna().astype(int)
    df_clean['email_address'] = df_clean['email_address'].fillna('unverified_email@internal.placeholder')

    df_clean['customer_age'] = df_clean['customer_age'].astype(float)
    valid_ages = (df_clean['customer_age'] >= 18) & (df_clean['customer_age'] <= 100)
    df_clean.loc[~valid_ages, 'customer_age'] = df_clean.loc[valid_ages, 'customer_age'].median()

    df_clean['valid_phone_flag'] = (df_clean['contact_phone'].astype(str).str.strip().str.len() == 10).astype(int)
    df_clean.to_csv('customers_clean.csv', index=False)

    print(f"\nCleaning Complete: Sanitized {len(df_clean)} records saved to customers_clean.csv")

def plot_learning_curve():
    X, y = make_classification(n_samples=2500, n_features=20, n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    cohorts = [50, 100, 250, 500, 1000, len(X_train)]
    scores = []
    for n in cohorts:
        rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train[:n], y_train[:n])
        scores.append(accuracy_score(y_test, rf.predict(X_test)))

    plt.figure(figsize=(8, 4.5), dpi=300)
    plt.plot(cohorts, scores, marker='o', color='#1f77b4', linewidth=2)
    plt.title("Empirical Learning Curve: Accuracy vs Sample Size")
    plt.xlabel("Training Samples")
    plt.ylabel("Test Accuracy")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("sample_size_curve.png", bbox_inches='tight')
    plt.close()
    print("Saved sample_size_curve.png")

if __name__ == '__main__':
    run_data_audit()
    plot_learning_curve()
