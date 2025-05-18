import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    df_clean = df.dropna(subset=[target])
    n_after = len(df_clean)
    print(f"> Filas con {target}=NaN eliminadas: {n_before - n_after}")

    # Separar características y variable objetivo
    X = df_clean.drop(columns=[target])
    y = df_clean[target]

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

        # Gráficas específicas para GradientBoosting
        if name == 'GradientBoosting':
            # Residuals plot
            residuals = y_test - y_pred
            plt.figure()
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(0, color='red', linewidth=1)
            plt.xlabel('Predicted values')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Predicted (GB)')
            plt.savefig('results/residuals_gb.png')
            plt.close()

            # Feature importance
            feat_importances = best.named_steps['model'].feature_importances_
            plt.figure()
            plt.bar(range(len(feat_importances)), feat_importances)
            plt.xlabel('Feature index')
            plt.ylabel('Importance')
            plt.title('Feature Importances (GB)')
            plt.savefig('results/feat_imp_gb.png')
            plt.close()

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
    # 1. Cargar datos
    df = load_data()

    # 2. EDA: gráficas de la semana 2
    os.makedirs('results', exist_ok=True)
    # Histograma de tiempos de entrega
    plt.figure()
    plt.hist(df['delivery_time'].dropna(), bins=50)
    plt.xlabel('Delivery Time (days)')
    plt.ylabel('Frequency')
    plt.title('Histograma de tiempos de entrega')
    plt.savefig('results/hist_delivery_time.png')
    plt.close()
    # Histograma de review_score
    plt.figure()
    plt.hist(df['review_score'].dropna(), bins=df['review_score'].nunique())
    plt.xlabel('Review Score')
    plt.ylabel('Frequency')
    plt.title('Histograma de puntuaciones de reseña')
    plt.savefig('results/hist_review_score.png')
    plt.close()
    # Bar chart top estados
    if 'customer_state' in df.columns:
        state_counts = df['customer_state'].value_counts().head(10)
        plt.figure()
        plt.bar(state_counts.index, state_counts.values)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Estado')
        plt.ylabel('Número de clientes')
        plt.title('Top 10 estados por número de clientes')
        plt.tight_layout()
        plt.savefig('results/mapa_clientes.png')
        plt.close()

    # 3. Ejecutar modelos y generar gráficas de regresión
    run_models(df)


if __name__ == '__main__':
    main()
