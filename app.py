# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import logging
from pathlib import Path

from db import save_prediction_to_db
from python_files.test_process_lstm import (
    filter_and_encode_test_data,
    get_test_sequences,
    predict_and_inverse_transform,
    load_lstm_model,
    load_encoders_and_scaler,
)
from python_files.predict import (
    load_model as load_catboost_model,
    run_prediction,
    prepare_input_for_prediction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Sales Prediction API",
    description="Predicts monthly item sales using CatBoost or LSTM models.",
    version="1.0.0"
)

class PredictRequest(BaseModel):
    shop_id: int = Field(..., example=125)
    item_id: int = Field(..., example=116790)
    next_month: str = Field(..., example="2015-10-01")
    model_type: str = Field(..., example="lstm")  # or "lstm"


catboost_model = load_catboost_model('models/catboost_model.pkl')
lstm_model = load_lstm_model('models/lstm_model.h5')
le_shop, le_item, le_cat, scaler = load_encoders_and_scaler(
    'le_shop.pkl', 'le_item.pkl', 'le_cat.pkl', 'scaler.pkl'
)

DATA_DIR = Path(__file__).resolve().parent / "data"
df_monthly_lstm = pd.read_csv(DATA_DIR / 'df_monthly_lstm.csv')
df_monthly_lstm_enc2 = pd.read_csv(DATA_DIR / 'df_monthly_lstm_enc2.csv')
train_pairs = set(zip(df_monthly_lstm['shop_id'], df_monthly_lstm['item_id']))

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        logging.info("Received request: %s", request.dict())

        if request.model_type == 'catboost':
            logging.info("Running CatBoost prediction")
            input_df = pd.DataFrame([request.dict()])
            features_df = prepare_input_for_prediction(input_df)
            preds = run_prediction(catboost_model, features_df)

            save_prediction_to_db(request.shop_id, request.item_id, request.next_month, float(preds[0]))

            return {"predicted_sales": float(preds[0])}

        elif request.model_type == 'lstm':
            logging.info("Running LSTM prediction")
            df_test = pd.DataFrame([{
                'shop': request.shop_id,
                'item': request.item_id,
                'next_month': request.next_month
            }])

            df_test_filtered = filter_and_encode_test_data(
                df_test, train_pairs, le_shop, le_item, le_cat, df_monthly_lstm
            )

            feature_cols = ['amount', 'sales_log', 'price_log', 'shop_id_enc', 'item_id_enc', 'item_category_enc']
            window_size = 12

            X_test_sequences, valid_test_indices = get_test_sequences(
                df_test_filtered, df_monthly_lstm_enc2, feature_cols, window_size
            )

            if len(X_test_sequences) == 0:
                raise HTTPException(status_code=400, detail="Not enough history to make LSTM prediction.")

            sales_pred_actual = predict_and_inverse_transform(
                lstm_model, X_test_sequences, scaler, ['amount', 'price_log', 'sales_log']
            )

            return {"predicted_sales": float(sales_pred_actual[0])}

        else:
            raise HTTPException(status_code=400, detail="Unsupported model_type. Use 'catboost' or 'lstm'.")

    except Exception as e:
        logging.error("Prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
