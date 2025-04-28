import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(path='data/preprocessed_regression.csv'):
    """Carga el dataset preprocesado."""
    print(f"Cargando datos desde {path}...")
    df = pd.read_csv(path)
    print(f"> Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.")
    return df

def run_models(df, target='delivery_time', test_size=0.2, random_state=42):
    # —> Eliminar filas donde target sea NaN
    n_before = len(df)
    df = df.dropna(subset=[target])
    n_after = len(df)
    print(f"> Filas con {target}=NaN eliminadas: {n_before - n_after}")

    # Separar características y variable objetivo
    X = df.drop(columns=[target])
    y = df[target]

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Identificar columnas numéricas y categóricas
    numeric_feats = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_feats = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Pipelines de preprocesamiento
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_feats),
        ('cat', categorical_pipeline, categorical_feats)
    ])

    # Definir modelos y grid de hiperparámetros
    models = {
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(random_state=random_state),
        'RandomForest': RandomForestRegressor(random_state=random_state),
        'GradientBoosting': GradientBoostingRegressor(random_state=random_state)
    }
    param_grids = {
        'LinearRegression': {},
        'DecisionTree': {
            'model__max_depth': [None, 5, 10],
            'model__min_samples_split': [2, 5, 10]
        },
        'RandomForest': {
            'model__n_estimators': [100],
            'model__max_depth': [None, 10],
            'model__min_samples_split': [2, 5]
        },
        'GradientBoosting': {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1],
            'model__max_depth': [3, 5]
        }
    }

    # Crear carpeta de resultados
    os.makedirs('results', exist_ok=True)

    results = []
    for name, estimator in models.items():
        print(f"\n=== {name} ===")
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', estimator)
        ])
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grids[name],
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)

        best = grid.best_estimator_
        y_pred = best.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"Mejores parámetros: {grid.best_params_}")
        print(f"MAE:   {mae:.4f}")
        print(f"RMSE:  {rmse:.4f}")
        print(f"R²:    {r2:.4f}")

        results.append({
            'model': name,
            'best_params': grid.best_params_,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })

    # Guardar resultados en CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/regression_results.csv', index=False)
    print("\nResultados guardados en 'results/regression_results.csv'.")

def main():
    df = load_data()
    run_models(df)

if __name__ == '__main__':
    main()