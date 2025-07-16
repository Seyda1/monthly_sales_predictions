import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

def load_encoders_and_scaler(le_shop_path, le_item_path, le_cat_path, scaler_path):
    le_shop = joblib.load(le_shop_path)
    le_item = joblib.load(le_item_path)
    le_cat = joblib.load(le_cat_path)
    scaler = joblib.load(scaler_path)
    return le_shop, le_item, le_cat, scaler

def filter_and_encode_test_data(df_test, train_pairs, le_shop, le_item, le_cat, df_monthly_lstm):
   
    df_test['pair'] = list(zip(df_test['shop'], df_test['item']))
    df_test_filtered = df_test[df_test['pair'].isin(train_pairs)].copy()
    df_test_filtered.drop(columns=['pair'], inplace=True)

    
    df_test_filtered.rename(columns={'shop': 'shop_id', 'item': 'item_id'}, inplace=True)

    
    df_test_filtered['shop_id_enc'] = le_shop.transform(df_test_filtered['shop_id'])
    df_test_filtered['item_id_enc'] = le_item.transform(df_test_filtered['item_id'])


    item_category_lookup = df_monthly_lstm[['item_id', 'item_category_id']].drop_duplicates()
    df_test_filtered = df_test_filtered.merge(item_category_lookup, on='item_id', how='left')
    df_test_filtered['item_category_enc'] = le_cat.transform(df_test_filtered['item_category_id'])
    df_test_filtered.drop(columns=['item_category_id'], inplace=True)

    return df_test_filtered

def get_test_sequences(test_df, train_df, feature_cols, window_size):
    sequences = []
    valid_indices = []

    for idx, row in test_df.iterrows():
        shop = row['shop_id_enc']
        item = row['item_id_enc']
        item_cat = row['item_category_enc']  

      
        hist = train_df[(train_df['shop_id_enc'] == shop) & (train_df['item_id_enc'] == item)]
        hist = hist.sort_values('date')

        if len(hist) < window_size:
            continue

        last_window = hist.iloc[-window_size:].copy()

       
        last_window['shop_id_enc'] = shop
        last_window['item_id_enc'] = item
        last_window['item_category_enc'] = item_cat

        seq = last_window[feature_cols].values
        sequences.append(seq)
        valid_indices.append(idx)

    return np.array(sequences), valid_indices

def load_lstm_model(model_path):
    import tensorflow.keras.losses
    model = load_model(model_path, custom_objects={'mse': tensorflow.keras.losses.MeanSquaredError()})
    return model

def predict_and_inverse_transform(model, X_test, scaler, numeric_features):
    sales_log_idx = numeric_features.index('sales_log')

    predictions = model.predict(X_test).flatten()
    dummy = np.zeros((len(predictions), len(numeric_features)))
    dummy[:, sales_log_idx] = predictions

    inv_scaled = scaler.inverse_transform(dummy)
    sales_log_inv = inv_scaled[:, sales_log_idx]
    sales_pred_actual = np.expm1(sales_log_inv)

    return sales_pred_actual


def main():
   
    df_test = pd.read_csv('df_test.csv')
    df_monthly_lstm = pd.read_csv('df_monthly_lstm.csv')
    df_monthly_lstm_enc = pd.read_csv('df_monthly_lstm_enc2.csv') 
    
 
    le_shop, le_item, le_cat, scaler = load_encoders_and_scaler(
        'le_shop.pkl', 'le_item.pkl', 'le_cat.pkl', 'scaler.pkl')

    train_pairs = set(zip(df_monthly_lstm['shop_id'], df_monthly_lstm['item_id']))

  
    df_test_filtered = filter_and_encode_test_data(df_test, train_pairs, le_shop, le_item, le_cat, df_monthly_lstm)

    
    feature_cols = ['amount', 'sales_log', 'price_log', 'shop_id_enc', 'item_id_enc', 'item_category_enc']
    window_size = 12

    X_test_sequences, valid_test_indices = get_test_sequences(df_test_filtered, df_monthly_lstm_enc, feature_cols, window_size)


    model = load_lstm_model('lstm_model.h5')


    sales_predictions = predict_and_inverse_transform(model, X_test_sequences, scaler, ['amount', 'price_log', 'sales_log'])

    df_test_filtered = df_test_filtered.iloc[valid_test_indices].copy()
    df_test_filtered['predicted_sales'] = sales_predictions

   