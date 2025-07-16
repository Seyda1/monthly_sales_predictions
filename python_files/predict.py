import joblib
import pandas as pd
from pathlib import Path
from catboost import Pool



def load_model(model_path: str):
    """
    Load the trained model from disk.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        model: Loaded model object.
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise
    

def run_prediction(model, input_df: pd.DataFrame) -> pd.Series:
    try:
        MODEL_FEATURES = [
            'shop_id', 'item_id',
            'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
            'month', 'year', 'quarter',
            'is_month_start', 'is_month_end', 'season',
            'rolling_mean_3', 'trend_1_2', 'lag_1_ratio_2'
        ]

       
        input_df = input_df[MODEL_FEATURES].copy()
        cat_features = ['shop_id', 'item_id', 'month', 'year', 'quarter',
                        'is_month_start', 'is_month_end', 'season']
        for col in cat_features:
            input_df[col] = input_df[col].astype('category')

        pool = Pool(data=input_df, cat_features=cat_features)
        preds = model.predict(pool)
        return preds

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

    
    
def prepare_input_for_prediction(raw_input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare input features for prediction from raw input dataframe by:
    - Loading historical monthly sales data
    - Creating lag, rolling mean, trend, and date features on historical data
    - Merging input test data with last available dates to get corresponding historical features
    - Preparing features for the next month for prediction
    - Cleaning up unnecessary columns
    
    Args:
        raw_input_df (pd.DataFrame): Raw input dataframe with at least columns ['shop_id', 'item_id', 'date'].
        
    Returns:
        pd.DataFrame: Dataframe containing all features required by the prediction model,
                      ready for input to the trained model.
                      
    Raises:
        FileNotFoundError: If 'monthly_sales.csv' file cannot be found or loaded.
        ValueError: If input dataframe does not contain required columns or if preprocessing fails.
        Exception: For any other unforeseen errors during feature preparation.
    """

    try:
        from python_files.testpreprocess import (
            prepare_monthly_sales_features_test,
            merge_test_with_last_dates,
            create_last_features,
            prepare_test_next_month_features,
        )
        import pandas as pd

    

        df_new = prepare_monthly_sales_features_test()
        if df_new.empty:
            raise ValueError("Historical monthly sales data is empty.")
       
        required_columns = {'shop_id', 'item_id', 'next_month'}
        if not required_columns.issubset(raw_input_df.columns):
            missing = required_columns - set(raw_input_df.columns)
            raise ValueError(f"Input data is missing required columns: {missing}")
        test_df = raw_input_df.copy()
        test_df['next_month'] = pd.to_datetime(test_df['next_month'])
        test_pairs_with_dates = merge_test_with_last_dates(test_df, df_new)
      
        last_features = create_last_features(df_new, test_pairs_with_dates)
        test_next_month = prepare_test_next_month_features(last_features)

       
        for col in ['sales', 'last_date', 'next_month', 'date']:
            if col in test_next_month.columns:
                test_next_month = test_next_month.drop(columns=[col])

     
        test_next_month = test_next_month.reset_index(drop=True)
        return test_next_month

    except FileNotFoundError as fnf_error:
        print(f"File not found error: {fnf_error}")
        raise

    except ValueError as val_error:
        print(f"Value error in input data or processing: {val_error}")
        raise

    except Exception as e:
        print(f"Unexpected error during feature preparation: {e}")
        raise

