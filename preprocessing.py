import pandas as pd
import numpy as np
import os
import glob

INPUT_DIR = 'data'
OUTPUT_DIR = 'data_cleaned'

# Función para cargar datasets crudos desde INPUT_DIR
def load_raw_datasets(input_dir):
    """
    Carga todos los datasets CSV desde el directorio de entrada especificado.
    """
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    dataframes = {}
    print(f"Cargando datasets desde: {input_dir}")
    for f in all_files:
        try:
            base_name = os.path.basename(f)
            name = base_name.replace('.csv', '').replace('olist_', '').replace('_dataset', '').replace('_compressed', '')
            dataframes[name] = pd.read_csv(f)
            print(f"  ✓ {name} ({base_name}) cargado.")
        except Exception as e:
            print(f"  ✗ Error al cargar {base_name}: {e}")
    return dataframes

# Función para guardar datasets limpios en OUTPUT_DIR
def save_cleaned_datasets(dataframes, output_dir):
    """
    Guarda los dataframes procesados en el directorio de salida especificado.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGuardando datasets procesados en: {output_dir}")
    for name, df in dataframes.items():
        if name == 'geolocation':
             filename = f"olist_{name}_dataset_compressed.csv"
        elif name == 'product_category_name_translation':
             filename = f"{name}.csv"
        else:
            filename = f"olist_{name}_dataset.csv" 
            
        output_path = os.path.join(output_dir, filename)
        try:
            df.to_csv(output_path, index=False)
            print(f"  ✓ {name} guardado como {filename}.")
        except Exception as e:
            print(f"  ✗ Error al guardar {name} como {filename}: {e}")


def preprocess_dataframes(dataframes):
    print("\n" + "="*80)
    print(" "*30 + "INICIANDO PREPROCESAMIENTO DE DATOS")
    print("="*80)

    print("\n1. Convirtiendo columnas de fecha...")
    date_cols_orders = [
        'order_purchase_timestamp', 'order_approved_at',
        'order_delivered_carrier_date', 'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    if 'orders' in dataframes:
        df = dataframes['orders']
        for col in date_cols_orders:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f"  - orders.{col}: convertida.")
        dataframes['orders'] = df

    date_cols_reviews = ['review_creation_date', 'review_answer_timestamp']
    if 'order_reviews' in dataframes:
        df = dataframes['order_reviews']
        for col in date_cols_reviews:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f"  - order_reviews.{col}: convertida.")
        dataframes['order_reviews'] = df

    print("\n2. Eliminando duplicados...")
    for name, df in dataframes.items():
        initial_rows = len(df)
        id_col_map = {
            'customers': 'customer_id',
            'sellers': 'seller_id',
            'products': 'product_id',
        }
        if name in id_col_map:
            df.drop_duplicates(subset=[id_col_map[name]], inplace=True)
        else:
            df.drop_duplicates(inplace=True)
        
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            print(f"  - {name}: {rows_removed} filas duplicadas eliminadas.")
        dataframes[name] = df

    if 'products' in dataframes and 'product_category_name_translation' in dataframes:
        print("\n3. Merging de nombres de categorías de productos (Inglés)...")
        products = dataframes['products']
        categories = dataframes['product_category_name_translation']
        
        categories.drop_duplicates(subset=['product_category_name'], inplace=True)

        products = pd.merge(
            products, 
            categories, 
            on='product_category_name', 
            how='left'
        )
        
        products['product_category_name_english'].fillna('Unknown', inplace=True)
        print(f"  - Nombres en inglés añadidos a 'products'. Categorías sin traducción: {products['product_category_name_english'].eq('Unknown').sum()}")
        dataframes['products'] = products


    print("\n4. Manejando valores faltantes (ejemplo en 'products')...")
    if 'products' in dataframes:
        products = dataframes['products']
        missing_before = products.isnull().sum()

        num_cols_to_impute = [
            'product_name_lenght', 'product_description_lenght', 
            'product_photos_qty', 'product_weight_g', 
            'product_length_cm', 'product_height_cm', 'product_width_cm'
        ]
        for col in num_cols_to_impute:
            if col in products.columns:
                 if products[col].isnull().any():
                    median_val = products[col].median()
                    products[col].fillna(median_val, inplace=True)
                    print(f"  - products.{col}: NaN rellenados con mediana ({median_val}). Faltantes antes: {missing_before.get(col, 0)}")
            else:
                print(f"  - Advertencia: La columna {col} no existe en el dataframe 'products'.")

        if 'product_category_name' in products.columns and products['product_category_name'].isnull().any():
             products['product_category_name'].fillna('Unknown', inplace=True)
             print(f"  - products.product_category_name: NaN rellenados con 'Unknown'. Faltantes antes: {missing_before.get('product_category_name', 0)}")
             
        if 'product_category_name_english' in products.columns and products['product_category_name_english'].isnull().any():
             products['product_category_name_english'].fillna('Unknown', inplace=True)
             print(f"  - products.product_category_name_english: NaN rellenados con 'Unknown'. Faltantes antes: {missing_before.get('product_category_name_english', 0)}")


        dataframes['products'] = products
        
        if 'product_category_name_translation' in dataframes:
            del dataframes['product_category_name_translation']
            print("\n  - Dataframe 'product_category_name_translation' eliminado después del merge.")

    print("\n" + "="*80)
    print(" "*30 + "PREPROCESAMIENTO COMPLETADO")
    print("="*80 + "\n")

    return dataframes


if __name__ == "__main__":
    raw_dataframes = load_raw_datasets(INPUT_DIR)
    
    if raw_dataframes:
        cleaned_dataframes = preprocess_dataframes(raw_dataframes.copy())
        
        save_cleaned_datasets(cleaned_dataframes, OUTPUT_DIR)
    else:
        print("No se cargaron datasets. Terminando script de preprocesamiento.") 