# classification_week3.py

"""
Semana 3: Implementación de Modelos de Clasificación
Este script carga el dataset preprocesado y entrena múltiples clasificadores
para predecir la satisfacción del cliente (binary: satisfecho/no satisfecho).
"""

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

def load_data(path='data/preprocessed_regression.csv'):
    """Carga el CSV preprocesado y devuelve el DataFrame."""
    print(f"Cargando datos desde {path}...")
    df = pd.read_csv(path)
    print(f"> Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.")
    return df

def prepare_classification_df(df, threshold=4):
    """
    Crea la variable objetivo binaria 'satisfied':
      1 si review_score >= threshold, 0 de lo contrario.
    Elimina filas con NaN en review_score.
    """
    n_before = len(df)
    df = df.dropna(subset=['review_score'])
    df['satisfied'] = (df['review_score'] >= threshold).astype(int)
    df = df.drop(columns=['review_score', 'order_id'])
    n_after = len(df)
    print(f"> Filas sin review_score eliminadas: {n_before - n_after}")
    return df

def run_classification(df, test_size=0.2, random_state=42):
    """Entrena varios modelos de clasificación con GridSearchCV y evalúa su desempeño."""
    # Separar X e y
    X = df.drop(columns=['satisfied'])
    y = df['satisfied']

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Identificar numéricas y categóricas
    num_feats = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_feats = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Pipelines de preprocesamiento
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_feats),
        ('cat', cat_pipeline, cat_feats)
    ])

    # Definir modelos y grid de hiperparámetros reducida
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=random_state),
        'DecisionTree': DecisionTreeClassifier(random_state=random_state),
        'RandomForest': RandomForestClassifier(random_state=random_state),
        'GradientBoosting': GradientBoostingClassifier(random_state=random_state),
        'SVM': SVC(probability=True, random_state=random_state)
    }
    param_grids = {
        'LogisticRegression': {'model__C': [0.1, 1.0, 10.0]},
        'DecisionTree': {'model__max_depth': [None, 5, 10], 'model__min_samples_split': [2, 5]},
        'RandomForest': {'model__n_estimators': [100], 'model__max_depth': [None, 10]},
        'GradientBoosting': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1]},
        'SVM': {'model__C': [0.1, 1.0], 'model__kernel': ['rbf', 'linear']}
    }

    os.makedirs('results', exist_ok=True)
    results = []

    for name, estimator in models.items():
        print(f"\n== {name} ==")
        pipe = Pipeline([('preprocessor', preprocessor), ('model', estimator)])
        grid = GridSearchCV(
            estimator=pipe, param_grid=param_grids[name],
            cv=3, scoring='f1', n_jobs=-1, verbose=1
        )
        grid.fit(X_train, y_train)

        best = grid.best_estimator_
        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        print("Mejores parámetros:", grid.best_params_)
        print("Métricas:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  ROC AUC:   {roc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4))

        results.append({
            'model': name,
            'best_params': grid.best_params_,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc
        })

    # Guardar resultados
    res_df = pd.DataFrame(results)
    res_df.to_csv('results/classification_results.csv', index=False)
    print("\n>> Resultados guardados en 'results/classification_results.csv'")

def main():
    # 1. Cargar y preparar datos
    df = load_data()
    df_clf = prepare_classification_df(df)

    # 2. Entrenar y evaluar
    run_classification(df_clf)

if __name__ == '__main__':
    main()
