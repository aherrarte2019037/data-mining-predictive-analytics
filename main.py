"""
Análisis Exploratorio de Datos
Universidad del Valle de Guatemala - CC3074 - Minería de Datos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import missingno as msno  
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
warnings.filterwarnings('ignore')

# Configuraciones de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)

# Función para cargar todos los datasets
def load_datasets():
    """
    Carga todos los datasets de Olist y devuelve un diccionario con los dataframes
    """
    files = [
        'data/olist_customers_dataset.csv',
        'data/olist_geolocation_dataset_compressed.csv',
        'data/olist_order_items_dataset.csv',
        'data/olist_order_payments_dataset.csv',
        'data/olist_order_reviews_dataset.csv',
        'data/olist_orders_dataset.csv',
        'data/olist_products_dataset.csv',
        'data/olist_sellers_dataset.csv',
        'data/product_category_name_translation.csv'
    ]
    
    dataframes = {}
    
    for file in files:
        try:
            # Extraer nombre corto para el diccionario
            base_name = file.split('/')[-1] # Get filename only
            name = base_name.replace('.csv', '').replace('olist_', '').replace('_dataset', '').replace('_compressed', '')
            
            # Cargar el dataset
            dataframes[name] = pd.read_csv(file)
            
            print(f"✓ {name} cargado correctamente - Shape: {dataframes[name].shape}")
        except Exception as e:
            print(f"✗ Error al cargar {file}: {e}")
    
    return dataframes

# 1. EVALUACIÓN DE CALIDAD DE DATOS

def evaluate_data_quality(dataframes, image_dir):
    """
    Realiza una evaluación completa de la calidad de datos para todos los dataframes
    y genera un informe detallado
    """
    print("\n" + "="*80)
    print(" "*30 + "EVALUACIÓN DE CALIDAD DE DATOS")
    print("="*80)
    
    quality_report = {}
    
    for name, df in dataframes.items():
        print(f"\n\n{'-'*40}")
        print(f"DATASET: {name.upper()}")
        print(f"{'-'*40}")
        
        # 1.1 Información básica
        print(f"\nDimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
        
        # 1.2 Tipos de datos
        print("\nTipos de datos:")
        print(df.dtypes)
        
        # 1.3 Valores únicos para cada columna
        print("\nValores únicos por columna:")
        for col in df.columns:
            unique_count = df[col].nunique()
            print(f"  - {col}: {unique_count} valores únicos")
        
        # 1.4 Valores faltantes
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        
        print("\nValores faltantes:")
        missing_data = pd.DataFrame({
            'Cantidad': missing_values,
            'Porcentaje (%)': missing_percentage.round(2)
        })
        print(missing_data[missing_data['Cantidad'] > 0])
        
        # 1.5 Detectar valores duplicados
        duplicates = df.duplicated().sum()
        print(f"\nFilas duplicadas: {duplicates} ({(duplicates/len(df)*100):.2f}%)")
        
        # Guardar resultados en el reporte
        quality_report[name] = {
            'shape': df.shape,
            'dtypes': df.dtypes,
            'missing_values': missing_data[missing_data['Cantidad'] > 0],
            'duplicates': duplicates
        }
        
        # 1.6 Visualizar valores faltantes
        plt.figure(figsize=(12, 6))
        msno.matrix(df, figsize=(12, 6))
        plt.title(f'Matriz de valores faltantes - {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, f'missing_values_{name}.png'))
        plt.close()
        
        # 1.7 Detectar outliers en columnas numéricas
        print("\nAnalisis de outliers en columnas numéricas:")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                
                if len(outliers) > 0:
                    print(f"  - {col}: {len(outliers)} outliers ({(len(outliers)/len(df)*100):.2f}%)")
                    print(f"    Rango: [{df[col].min()}, {df[col].max()}]")
                    print(f"    IQR: {iqr}")
                    print(f"    Límites: [{lower_bound}, {upper_bound}]")
        
    return quality_report

# 2. RESÚMENES ESTADÍSTICOS

def generate_statistical_summaries(dataframes, image_dir):
    """
    Resúmenes estadísticos para todos los dataframes
    """
    print("\n" + "="*80)
    print(" "*30 + "RESÚMENES ESTADÍSTICOS")
    print("="*80)
    
    for name, df in dataframes.items():
        print(f"\n\n{'-'*40}")
        print(f"DATASET: {name.upper()}")
        print(f"{'-'*40}")
        
        # 2.1 Estadísticas descriptivas para columnas numéricas
        numeric_cols = df.select_dtypes(include=['int64', 'float64'])
        
        if not numeric_cols.empty:
            print("\nEstadísticas descriptivas para variables numéricas:")
            stats = numeric_cols.describe().T
            stats['CV (%)'] = (stats['std'] / stats['mean'] * 100).round(2)  # Coeficiente de variación
            print(stats)
            
            stats.to_csv(os.path.join(image_dir, f'stats_{name}.csv'))
            
            # 2.2 Visualización de distribuciones
            if len(numeric_cols.columns) > 0:
                n_cols = min(3, len(numeric_cols.columns))
                n_rows = (len(numeric_cols.columns) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
                axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
                
                for i, col in enumerate(numeric_cols.columns):
                    if i < len(axes):
                        sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
                        axes[i].set_title(f'Distribución de {col}')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Frecuencia')
                
                for j in range(i + 1, len(axes)):
                    axes[j].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(os.path.join(image_dir, f'distributions_{name}.png'))
                plt.close()
        
        # 2.3 Análisis de variables categóricas
        cat_cols = df.select_dtypes(include=['object'])
        
        if not cat_cols.empty and len(cat_cols.columns) > 0:
            print("\nAnálisis de variables categóricas:")
            
            for col in cat_cols.columns:
                value_counts = df[col].value_counts().reset_index()
                value_counts.columns = [col, 'Frecuencia']
                value_counts['Porcentaje'] = (value_counts['Frecuencia'] / len(df) * 100).round(2)
                
                if len(value_counts) > 10:
                    print(f"\n{col} - Top 10 categorías más frecuentes:")
                    print(value_counts.head(10))
                else:
                    print(f"\n{col} - Todas las categorías:")
                    print(value_counts)
                
                # Visualizar las 10 categorías más frecuentes
                plt.figure(figsize=(12, 6))
                top_categories = value_counts.head(10)
                sns.barplot(x='Frecuencia', y=col, data=top_categories)
                plt.title(f'Top 10 categorías más frecuentes - {col}')
                plt.tight_layout()
                plt.savefig(os.path.join(image_dir, f'categories_{name}_{col}.png'))
                plt.close()

# 3. VISUALIZACIÓN DE PATRONES CLAVE

def visualize_key_patterns(dataframes, image_dir):
    """
    Genera visualizaciones para identificar patrones clave en los datos
    """
    print("\n" + "="*80)
    print(" "*30 + "VISUALIZACIÓN DE PATRONES CLAVE")
    print("="*80)
    
    if 'orders' in dataframes:
        orders = dataframes['orders']
        
        if 'order_purchase_timestamp' in orders.columns:
            orders['purchase_year'] = orders['order_purchase_timestamp'].dt.year
            orders['purchase_month'] = orders['order_purchase_timestamp'].dt.month
            orders['purchase_day'] = orders['order_purchase_timestamp'].dt.day
            orders['purchase_weekday'] = orders['order_purchase_timestamp'].dt.weekday
            orders['purchase_hour'] = orders['order_purchase_timestamp'].dt.hour
            
            monthly_orders = orders.groupby(['purchase_year', 'purchase_month']).size().reset_index(name='count')
            monthly_orders['year_month'] = monthly_orders['purchase_year'].astype(str) + '-' + monthly_orders['purchase_month'].astype(str).str.zfill(2)
            
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='year_month', y='count', data=monthly_orders)
            plt.title('Tendencia de órdenes por mes')
            plt.xlabel('Año-Mes')
            plt.ylabel('Número de órdenes')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(image_dir, 'orders_by_month.png'))
            plt.close()
            
            plt.figure(figsize=(10, 6))
            weekday_map = {0:'Lunes', 1:'Martes', 2:'Miércoles', 3:'Jueves', 4:'Viernes', 5:'Sábado', 6:'Domingo'}
            orders['weekday_name'] = orders['purchase_weekday'].map(weekday_map)
            weekday_order = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
            
            weekday_counts = orders['weekday_name'].value_counts().reindex(weekday_order)
            sns.barplot(x=weekday_counts.index, y=weekday_counts.values)
            plt.title('Distribución de órdenes por día de la semana')
            plt.xlabel('Día de la semana')
            plt.ylabel('Número de órdenes')
            plt.tight_layout()
            plt.savefig(os.path.join(image_dir, 'orders_by_weekday.png'))
            plt.close()
            
            plt.figure(figsize=(12, 6))
            hourly_counts = orders['purchase_hour'].value_counts().sort_index()
            sns.barplot(x=hourly_counts.index, y=hourly_counts.values)
            plt.title('Distribución de órdenes por hora del día')
            plt.xlabel('Hora del día')
            plt.ylabel('Número de órdenes')
            plt.tight_layout()
            plt.savefig(os.path.join(image_dir, 'orders_by_hour.png'))
            plt.close()
    
    # 3.2 Análisis de satisfacción del cliente
    if 'order_reviews' in dataframes and 'orders' in dataframes:
        reviews = dataframes['order_reviews']
        orders = dataframes['orders']
        
        plt.figure(figsize=(10, 6))
        sns.countplot(x='review_score', data=reviews)
        plt.title('Distribución de puntuaciones de reseñas')
        plt.xlabel('Puntuación')
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, 'review_scores_distribution.png'))
        plt.close()
        
        if 'review_creation_date' in reviews.columns and 'order_purchase_timestamp' in orders.columns:
            order_reviews = pd.merge(orders, reviews, on='order_id', how='inner')
            
            order_reviews['days_to_review'] = (order_reviews['review_creation_date'] - order_reviews['order_purchase_timestamp']).dt.days
            
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='review_score', y='days_to_review', data=order_reviews)
            plt.title('Relación entre puntuación y tiempo para escribir la reseña')
            plt.xlabel('Puntuación')
            plt.ylabel('Días entre compra y reseña')
            plt.tight_layout()
            plt.savefig(os.path.join(image_dir, 'review_score_vs_response_time.png'))
            plt.close()
    
    # 3.3 Análisis de precios y categorías de productos
    if 'products' in dataframes and 'order_items' in dataframes and 'product_category_name_translation' in dataframes:
        products = dataframes['products']
        order_items = dataframes['order_items']
        categories = dataframes['product_category_name_translation']
        
        products_with_categories = pd.merge(products, categories, on='product_category_name', how='left')
        
        product_sales = pd.merge(order_items, products_with_categories, on='product_id', how='inner')
        
        price_by_category = product_sales.groupby('product_category_name_english')['price'].agg(['mean', 'median', 'min', 'max', 'count']).reset_index()
        price_by_category = price_by_category.sort_values('count', ascending=False).head(15)  # Top 15 categorías por volumen
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x='mean', y='product_category_name_english', data=price_by_category)
        plt.title('Precio promedio por categoría de producto (Top 15 categorías)')
        plt.xlabel('Precio promedio (R$)')
        plt.ylabel('Categoría')
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, 'price_by_category.png'))
        plt.close()
        
        plt.figure(figsize=(12, 6))
        sns.histplot(product_sales['price'], bins=50, kde=True)
        plt.title('Distribución de precios')
        plt.xlabel('Precio (R$)')
        plt.ylabel('Frecuencia')
        plt.xlim(0, product_sales['price'].quantile(0.99))
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, 'price_distribution.png'))
        plt.close()
    
    # 3.4 Análisis geográfico
    if 'customers' in dataframes and 'sellers' in dataframes:
        customers = dataframes['customers']
        sellers = dataframes['sellers']
        
        plt.figure(figsize=(14, 8))
        customer_states = customers['customer_state'].value_counts()
        sns.barplot(x=customer_states.index, y=customer_states.values)
        plt.title('Distribución de clientes por estado')
        plt.xlabel('Estado')
        plt.ylabel('Número de clientes')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, 'customers_by_state.png'))
        plt.close()
        
        plt.figure(figsize=(14, 8))
        seller_states = sellers['seller_state'].value_counts()
        sns.barplot(x=seller_states.index, y=seller_states.values)
        plt.title('Distribución de vendedores por estado')
        plt.xlabel('Estado')
        plt.ylabel('Número de vendedores')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, 'sellers_by_state.png'))
        plt.close()
    
    # 3.5 Análisis de tiempos de entrega
    if 'orders' in dataframes:
        orders = dataframes['orders']
        
        if all(col in orders.columns and pd.api.types.is_datetime64_any_dtype(orders[col]) 
               for col in ['order_delivered_customer_date', 'order_purchase_timestamp', 
                         'order_estimated_delivery_date']):
            orders['delivery_time'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
            orders['estimated_delivery_time'] = (orders['order_estimated_delivery_date'] - orders['order_purchase_timestamp']).dt.days
            orders['delivery_difference'] = orders['estimated_delivery_time'] - orders['delivery_time']
            
            delivery_analysis = orders.dropna(subset=['delivery_time', 'estimated_delivery_time'])
            delivery_analysis = delivery_analysis[delivery_analysis['delivery_time'] >= 0]
            
            plt.figure(figsize=(12, 6))
            sns.histplot(delivery_analysis['delivery_time'], bins=30, kde=True)
            plt.axvline(delivery_analysis['delivery_time'].median(), color='red', linestyle='--', label=f'Mediana: {delivery_analysis["delivery_time"].median():.1f} días')
            plt.title('Distribución de tiempos de entrega')
            plt.xlabel('Tiempo de entrega (días)')
            plt.ylabel('Frecuencia')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(image_dir, 'delivery_time_distribution.png'))
            plt.close()
            
            plt.figure(figsize=(12, 6))
            sns.scatterplot(x='estimated_delivery_time', y='delivery_time', data=delivery_analysis, alpha=0.3)
            
            min_val = min(delivery_analysis['estimated_delivery_time'].min(), delivery_analysis['delivery_time'].min())
            max_val = max(delivery_analysis['estimated_delivery_time'].max(), delivery_analysis['delivery_time'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.title('Tiempo estimado vs. tiempo real de entrega')
            plt.xlabel('Tiempo estimado (días)')
            plt.ylabel('Tiempo real (días)')
            plt.tight_layout()
            plt.savefig(os.path.join(image_dir, 'estimated_vs_actual_delivery.png'))
            plt.close()

# 4. ANÁLISIS DE CORRELACIÓN DE CARACTERÍSTICAS

def analyze_correlations(dataframes, image_dir):
    """
    Realiza análisis de correlación entre variables para identificar relaciones importantes
    """
    print("\n" + "="*80)
    print(" "*30 + "ANÁLISIS DE CORRELACIÓN DE CARACTERÍSTICAS")
    print("="*80)
    
    # 4.1 Correlación en dataset de productos
    if 'products' in dataframes:
        products = dataframes['products']
        
        numeric_cols = products.select_dtypes(include=['int64', 'float64'])
        
        if not numeric_cols.empty and numeric_cols.shape[1] > 1:
            corr_matrix = numeric_cols.corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=.5)
            plt.title('Matriz de correlación - Características de productos')
            plt.tight_layout()
            plt.savefig(os.path.join(image_dir, 'product_correlation_matrix.png'))
            plt.close()
            
            corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
            corr_pairs = corr_pairs[corr_pairs < 1.0]
            
            print("\nCorrelaciones más fuertes entre características de productos:")
            print(corr_pairs.head(10)) 
            print("\nCorrelaciones más débiles entre características de productos:")
            print(corr_pairs.tail(10))  
    
    # 4.2 Correlación para análisis de tiempos de entrega
    if 'orders' in dataframes and 'order_items' in dataframes:
        orders = dataframes['orders']
        items = dataframes['order_items']
        
        if all(col in orders.columns and pd.api.types.is_datetime64_any_dtype(orders[col]) 
               for col in ['order_delivered_customer_date', 'order_purchase_timestamp', 
                         'order_estimated_delivery_date']):
            orders['delivery_time'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
            
            order_delivery = pd.merge(orders, items, on='order_id', how='inner')
            
            order_delivery['weekend_purchase'] = order_delivery['order_purchase_timestamp'].dt.weekday >= 5  # 5=Sábado, 6=Domingo
            
            delivery_vars = order_delivery[['delivery_time', 'price', 'freight_value', 'weekend_purchase']].copy()
            delivery_vars['weekend_purchase'] = delivery_vars['weekend_purchase'].astype(int)  # Convertir boolean a int
            
            delivery_corr = delivery_vars.corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(delivery_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=.5)
            plt.title('Correlación entre precio, flete y tiempo de entrega')
            plt.tight_layout()
            plt.savefig(os.path.join(image_dir, 'delivery_correlations.png'))
            plt.close()
    
    # 4.3 Correlación entre satisfacción y variables de servicio
    if all(k in dataframes for k in ['orders', 'order_reviews', 'order_items']):
        orders = dataframes['orders']
        reviews = dataframes['order_reviews']
        items = dataframes['order_items']
        
        if all(col in orders.columns and pd.api.types.is_datetime64_any_dtype(orders[col]) 
               for col in ['order_delivered_customer_date', 'order_purchase_timestamp', 
                         'order_estimated_delivery_date']):
            orders['delivery_time'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
            orders['delivery_delay'] = (orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']).dt.days
            
            order_satisfaction = pd.merge(orders, reviews[['order_id', 'review_score']], on='order_id', how='inner')
            
            items_agg = items.groupby('order_id').agg({
                'price': 'sum',
                'freight_value': 'mean',
                'order_item_id': 'count' 
            }).reset_index()
            
            items_agg.rename(columns={'order_item_id': 'item_count'}, inplace=True)
            
            order_satisfaction = pd.merge(order_satisfaction, items_agg, on='order_id', how='inner')
            
            satisfaction_vars = order_satisfaction[[
                'review_score', 'delivery_time', 'delivery_delay', 
                'price', 'freight_value', 'item_count'
            ]].dropna()
            
            satisfaction_corr = satisfaction_vars.corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(satisfaction_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=.5)
            plt.title('Correlación entre satisfacción del cliente y variables de servicio')
            plt.tight_layout()
            plt.savefig(os.path.join(image_dir, 'satisfaction_correlations.png'))
            plt.close()
            
            # Análisis específico de correlación entre demora y satisfacción
            if 'delivery_delay' in satisfaction_vars.columns:
                plt.figure(figsize=(12, 6))
                sns.boxplot(x='review_score', y='delivery_delay', data=satisfaction_vars)
                plt.axhline(y=0, color='red', linestyle='--')
                plt.title('Relación entre puntuación y retraso en la entrega')
                plt.xlabel('Puntuación')
                plt.ylabel('Retraso en días (negativo = entrega anticipada)')
                plt.tight_layout()
                plt.savefig(os.path.join(image_dir, 'review_score_vs_delivery_delay.png'))
                plt.close()

# Función principal
def main():
    """
    Función principal que ejecuta todo el análisis exploratorio de datos
    """
    print("\n" + "="*80)
    print(" "*20 + "ANÁLISIS EXPLORATORIO DE DATOS - PROYECTO OLIST")
    print("="*80 + "\n")
    
    # Create images directory
    image_dir = 'images'
    os.makedirs(image_dir, exist_ok=True)
    print(f"✓ Directorio '{image_dir}' creado o ya existente.")

    # 1. Cargar datasets
    dataframes = load_datasets()

    # Preprocessing: Convert date columns
    print("\nPreprocessing: Convirtiendo columnas de fecha...")
    date_cols_orders = [
        'order_purchase_timestamp', 'order_approved_at',
        'order_delivered_carrier_date', 'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    if 'orders' in dataframes:
        for col in date_cols_orders:
            if col in dataframes['orders'].columns:
                dataframes['orders'][col] = pd.to_datetime(dataframes['orders'][col], errors='coerce')
                print(f"  - orders.{col} convertida a datetime.")

    date_cols_reviews = ['review_creation_date', 'review_answer_timestamp']
    if 'order_reviews' in dataframes:
        for col in date_cols_reviews:
            if col in dataframes['order_reviews'].columns:
                dataframes['order_reviews'][col] = pd.to_datetime(dataframes['order_reviews'][col], errors='coerce')
                print(f"  - order_reviews.{col} convertida a datetime.")
    
    # 2. Evaluación de calidad de datos
    quality_report = evaluate_data_quality(dataframes, image_dir)
    
    # 3. Resúmenes estadísticos
    generate_statistical_summaries(dataframes, image_dir)
    
    # 4. Visualización de patrones clave
    visualize_key_patterns(dataframes, image_dir)
    
    # 5. Análisis de correlación
    analyze_correlations(dataframes, image_dir)
    
    print("\n" + "="*80)
    print(" "*25 + "ANÁLISIS EXPLORATORIO COMPLETADO")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()