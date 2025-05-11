import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning

# Ignorar avisos de convergencia de MLPClassifier
warnings.filterwarnings("ignore", category=ConvergenceWarning)

FIG_DIR = 'results/figures'


def load_data(path='data/preprocessed_regression.csv'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")
    return df


def prepare_classification_df(df, threshold=4):
    df = df.dropna(subset=['review_score'])
    df['satisfied'] = (df['review_score'] >= threshold).astype(int)
    df = df.drop(columns=['review_score', 'order_id'])
    return df


def build_preprocessor(X):
    num_feats = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_feats = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ('num', num_pipeline, num_feats),
        ('cat', cat_pipeline, cat_feats)
    ])


def evaluate_baseline(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    metrics = []
    os.makedirs('results', exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    for name, estimator in models.items():
        print(f"\n[BASELINE] Entrenando {name} sin CV")
        pipe = Pipeline([('preprocessor', preprocessor), ('model', estimator)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        metrics.append({'model': name, 'f1': f1})

    df_baseline = pd.DataFrame(metrics)
    df_baseline.to_csv('results/baseline_results.csv', index=False)

    # Gráfica F1 baseline
    plt.figure()
    plt.bar(df_baseline['model'], df_baseline['f1'])
    plt.ylabel('F1-score')
    plt.title('Baseline Models F1-score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/baseline_f1.png")
    plt.close()


def evaluate_cross_validation(X, y, preprocessor, cv=5):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    cv_results = []

    for name, estimator in models.items():
        print(f"\n[CV] Validando {name} con {cv}-fold CV")
        pipe = Pipeline([('preprocessor', preprocessor), ('model', estimator)])
        scores = cross_val_score(pipe, X, y, cv=kf, scoring='f1', n_jobs=-1)
        cv_results.append({'model': name, 'mean_f1': scores.mean(), 'std_f1': scores.std()})

    df_cv = pd.DataFrame(cv_results)
    df_cv.to_csv('results/cv_results.csv', index=False)

    # Gráfica CV F1
    plt.figure()
    plt.errorbar(df_cv['model'], df_cv['mean_f1'], yerr=df_cv['std_f1'], fmt='o-', capsize=5)
    plt.ylabel('F1-score')
    plt.title('Cross-Validation F1-score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cv_f1.png")
    plt.close()


def evaluate_neural_network(X_train, X_test, y_train, y_test, preprocessor):
    print("\n[NN] Entrenando MLPClassifier con GridSearchCV")
    mlp = MLPClassifier(random_state=42, max_iter=500, early_stopping=True)
    pipe = Pipeline([('preprocessor', preprocessor), ('model', mlp)])

    param_grid = {
        'model__hidden_layer_sizes': [(50,), (100,), (50,50)],
        'model__alpha': [0.0001, 0.001],
        'model__learning_rate_init': [0.001, 0.01]
    }
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    df_nn = pd.DataFrame([{'model': 'NeuralNetwork', 'f1': f1}])
    df_nn.to_csv('results/nn_results.csv', index=False)

    # Gráfica F1 NN vs GradientBoosting
    # Leer baseline para comparativa
    df_baseline = pd.read_csv('results/baseline_results.csv')
    gb_f1 = df_baseline.loc[df_baseline['model']=='GradientBoosting','f1'].values[0]

    plt.figure()
    plt.bar(['GradientBoosting','NeuralNetwork'], [gb_f1, f1])
    plt.ylabel('F1-score')
    plt.title('Comparación F1: GB vs NN')
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/nn_vs_gb_f1.png")
    plt.close()


def main():
    df = load_data()
    df = prepare_classification_df(df)
    X = df.drop(columns=['satisfied'])
    y = df['satisfied']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = build_preprocessor(X)

    evaluate_baseline(X_train, X_test, y_train, y_test, preprocessor)
    evaluate_cross_validation(X, y, preprocessor, cv=5)
    evaluate_neural_network(X_train, X_test, y_train, y_test, preprocessor)

if __name__ == '__main__':
    main()
