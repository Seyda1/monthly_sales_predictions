import sys
from pathlib import Path
import pandas as pd
import numpy as np

current_dir = Path().resolve()
python_files_dir = current_dir.parent / "Python_Files"
if str(python_files_dir) not in sys.path:
    sys.path.append(str(python_files_dir))

from python_files.config import CFG
from python_files.helpers import load_selected_data

def get_season(month: int) -> str:
    """
    Return season name based on month number.
    """
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'
    
    
def prepare_monthly_sales_features_test() -> pd.DataFrame:
    """
    Prepare monthly sales data with lag, rolling, trend, ratio and date features.

    Args:
        df (pd.DataFrame): Input monthly sales dataframe with columns ['shop_id', 'item_id', 'date', 'sales']

    Returns:
        pd.DataFrame: Dataframe enriched with lag, rolling mean, trend, ratio, and date features
    """
    try:
        df = pd.read_csv("/Users/seydaaybar/Desktop/ntt_data/data/monthly_sales_with_features.csv")

        return df
    
    except Exception as e:
        print(f"Error in prepare_monthly_sales_features: {e}")
        raise
    
    
def merge_test_with_last_dates(test_df: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """
    Merge test dataframe with last transaction dates per (shop_id, item_id).

    Args:
        test_df (pd.DataFrame): Test dataframe with ['shop_id', 'item_id']
        df_new (pd.DataFrame): Monthly sales dataframe with ['shop_id', 'item_id', 'date']

    Returns:
        pd.DataFrame: test_df enriched with 'last_date' column
    """
    try:
        last_dates = df_new.groupby(['shop_id', 'item_id'])['date'].max().reset_index()
        last_dates = last_dates.rename(columns={'date': 'last_date'})

        test_pairs_with_dates = pd.merge(test_df, last_dates, on=['shop_id', 'item_id'], how='left')

        global_max_date = df_new['date'].max()
        test_pairs_with_dates['last_date'] = test_pairs_with_dates['last_date'].fillna(global_max_date)

        test_pairs_with_dates['last_date'] = pd.to_datetime(test_pairs_with_dates['last_date'])

        return test_pairs_with_dates

    except Exception as e:
        print(f"Error in merge_test_with_last_dates: {e}")
        raise
    
def create_last_features(df_new: pd.DataFrame, test_pairs_with_dates: pd.DataFrame) -> pd.DataFrame:
    """
    Create last features dataframe by merging monthly sales and test pairs with last dates.

    Args:
        df_new (pd.DataFrame): Monthly sales dataframe with features
        test_pairs_with_dates (pd.DataFrame): Test pairs with 'last_date' column

    Returns:
        pd.DataFrame: last_features dataframe with filled missing values in lag and feature columns
    """
    try:
        df_new['date'] = pd.to_datetime(df_new['date'])
        test_pairs_with_dates['last_date'] = pd.to_datetime(test_pairs_with_dates['last_date'])

        lag_cols = [col for col in df_new.columns if col.startswith('lag_')]
        last_features = pd.merge(
            df_new,
            test_pairs_with_dates,
            left_on=['shop_id', 'item_id', 'date'],
            right_on=['shop_id', 'item_id', 'last_date'],
            how='right'
        )

        for col in lag_cols + ['rolling_mean_3', 'trend_1_2', 'lag_1_ratio_2']:
            if col in last_features.columns:
                last_features[col] = last_features[col].fillna(0)

        return last_features

    except Exception as e:
        print(f"Error in create_last_features: {e}")
        raise
    
def prepare_test_next_month_features(last_features: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for the next month based on last features dataframe.

    Args:
        last_features (pd.DataFrame): DataFrame with last month features

    Returns:
        pd.DataFrame: DataFrame with features for next month, cleaned and ready for prediction
    """
    try:
       
        test_next_month = last_features.copy()
   
        test_next_month["next_month"] = test_next_month["next_month"]
        test_next_month["next_month"] = pd.to_datetime(test_next_month["next_month"])
        test_next_month['month'] = test_next_month['next_month'].dt.month
        test_next_month['year'] = test_next_month['next_month'].dt.year
        test_next_month['quarter'] = test_next_month['next_month'].dt.quarter
        test_next_month['is_month_start'] = test_next_month['next_month'].dt.is_month_start.astype(int)
        test_next_month['is_month_end'] = test_next_month['next_month'].dt.is_month_end.astype(int)
        test_next_month['season'] = test_next_month['month'].apply(get_season)
        print("test_next_month", test_next_month.head())
      
        if 'sales' in test_next_month.columns:
            test_next_month = test_next_month.drop(columns=['sales'])

     
        if 'last_date' in test_next_month.columns:
            test_next_month = test_next_month.drop(columns=['next_month'])

        test_next_month = test_next_month.reset_index(drop=True)
        return test_next_month

    except Exception as e:
        print(f"Error in prepare_test_next_month_features: {e}")
        raise
    

def load_and_prepare_test_data():
    """
    Load all data using helpers, prepare test data and monthly sales features.

    Returns:
        tuple: (test_df, df_new, test_pairs_with_dates, last_features, test_next_month)
    """
    try:
        data = load_selected_data(CFG, ['df_test'])
        df_test = data['df_test']
        test_df = df_test.rename(columns={'shop': 'shop_id', 'item': 'item_id'})

        df_new = pd.read_csv('monthly_sales.csv')
        df_new = prepare_monthly_sales_features_test(df_new)

        test_pairs_with_dates = merge_test_with_last_dates(test_df=test_df, df_new=df_new)
        last_features = create_last_features(df_new=df_new, test_pairs_with_dates=test_pairs_with_dates)
        test_next_month = prepare_test_next_month_features(last_features=last_features)
        test_next_month = test_next_month.drop(columns=['next_month'], errors='ignore')
        return test_df, df_new, test_pairs_with_dates, last_features, test_next_month

    except Exception as e:
        print(f"Error in load_and_prepare_test_data: {e}")
        raise