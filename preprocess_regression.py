import os
import pandas as pd

def load_raw_data(data_dir='data'):
    orders = pd.read_csv(
        os.path.join(data_dir, 'olist_orders_dataset.csv'),
        parse_dates=[
            'order_purchase_timestamp',
            'order_approved_at',
            'order_delivered_carrier_date',
            'order_delivered_customer_date',
            'order_estimated_delivery_date'
        ]
    )
    items = pd.read_csv(os.path.join(data_dir, 'olist_order_items_dataset.csv'))
    products = pd.read_csv(os.path.join(data_dir, 'olist_products_dataset.csv'))
    reviews = pd.read_csv(
        os.path.join(data_dir, 'olist_order_reviews_dataset.csv'),
        parse_dates=['review_creation_date', 'review_answer_timestamp']
    )
    return orders, items, products, reviews

def engineer_features(orders, items, products, reviews):
    # 1) Merge
    df = (
        orders
        .merge(items, on='order_id', how='inner')
        .merge(products, on='product_id', how='left')
        .merge(
            reviews[['order_id', 'review_score', 'review_creation_date']],
            on='order_id', how='left'
        )
    )

    # 2) Feature engineering
    df['delivery_time'] = (df['order_delivered_customer_date']
                           - df['order_purchase_timestamp']).dt.days
    df['estimated_delivery_time'] = (df['order_estimated_delivery_date']
                                    - df['order_purchase_timestamp']).dt.days
    df['delivery_diff'] = df['estimated_delivery_time'] - df['delivery_time']
    df['review_delay'] = (df['review_creation_date']
                          - df['order_purchase_timestamp']).dt.days

    # 3) Agrupación por orden
    agg = df.groupby('order_id').agg({
        'delivery_time': 'first',
        'estimated_delivery_time': 'first',
        'delivery_diff': 'first',
        'review_score': 'mean',
        'review_delay': 'mean',
        'price': 'sum',
        'freight_value': 'mean',
        'product_weight_g': 'mean',
        'product_length_cm': 'mean',
        'product_height_cm': 'mean',
        'product_width_cm': 'mean',
        'order_item_id': 'count'  # número de ítems por orden
    }).reset_index()

    agg.rename(columns={'order_item_id': 'items_count'}, inplace=True)

    # 4) Limpieza básica: imputación simple en el DataFrame resultante
    #    (se puede ajustar más adelante)
    agg.fillna({
        'review_score': agg['review_score'].median(),
        'review_delay': agg['review_delay'].median()
    }, inplace=True)

    return agg

def main():
    # Carga tablas crudas
    orders, items, products, reviews = load_raw_data()
    print("Raw data cargada:")
    print(f" orders:   {orders.shape}")
    print(f" items:    {items.shape}")
    print(f" products: {products.shape}")
    print(f" reviews:  {reviews.shape}")

    # Genera dataset de modelado
    model_df = engineer_features(orders, items, products, reviews)
    print(f"\nDataset para modelado: {model_df.shape}")
    print(model_df.head())

    # Crea carpeta y guarda CSV
    os.makedirs('data', exist_ok=True)
    model_df.to_csv('data/preprocessed_regression.csv', index=False)
    print("\n>> 'data/preprocessed_regression.csv' creado con éxito.")

if __name__ == '__main__':
    main()
