# helpers.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def safe_read_csv(path):
    path = Path(path)
    if not path.exists():
        print(f"File not found: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df.drop(columns=["Unnamed: 0"], errors="ignore")
    except pd.errors.EmptyDataError:
        print(f"File is empty: {path}")
        return pd.DataFrame()
    except Exception as e:
        print(f" Error reading {path}: {e}")
        return pd.DataFrame()
    
def load_selected_data(cfg, keys: list) -> dict:
    """Loads only the requested datasets based on keys (e.g. ['df_transaction'])"""
    data_map = {
        'df_transaction': cfg.df_transaction_path,
        'df_test': cfg.df_test_path,
        'df_shop_list': cfg.df_shop_list_path,
        'df_item_list': cfg.df_item_list_path,
        'df_category_list': cfg.df_category_list_path,
        'df': cfg.df_path,
        'df_new': cfg.df_new_path
    }

    data = {}
    for key in keys:
        if key in data_map:
            data[key] = safe_read_csv(data_map[key])
        else:
            print(f"⚠️ Key '{key}' not recognized in config.")
    return data


def merge_transaction_data(df_transaction, df_item_list, df_category_list, df_shop_list)-> pd.DataFrame:
    """
    Merges transaction data with item, category, and shop lists.
    Args:
        df_transaction (DataFrame): DataFrame containing transaction data.
        df_item_list (DataFrame): DataFrame containing item list.
        df_category_list (DataFrame): DataFrame containing category list.
        df_shop_list (DataFrame): DataFrame containing shop list.
    Returns:
        DataFrame: Merged DataFrame containing transaction data with item, category, and shop information.
    """
    
    required_cols = {
        'df_transaction': ['item', 'shop'],
        'df_item_list': ['item_id', 'item_category_id'],
        'df_category_list': ['item_category_id'],
        'df_shop_list': ['shop_id']
    }
    dfs = {
        'df_transaction': df_transaction,
        'df_item_list': df_item_list,
        'df_category_list': df_category_list,
        'df_shop_list': df_shop_list
    }
    for name, cols in required_cols.items():
        missing = [col for col in cols if col not in dfs[name].columns]
        if missing:
            raise KeyError(f"Missing columns {missing} in {name}.")
    try:
        df_merged = df_transaction.merge(df_item_list, left_on="item", right_on="item_id", how="left")
        df_merged = df_merged.merge(df_category_list, on="item_category_id", how="left")
        df_merged = df_merged.merge(df_shop_list, left_on="shop", right_on="shop_id", how="left")
    except Exception as e:
        raise RuntimeError(f"Error during merging dataframes: {e}")
    
    return df_merged

def drop_columns(df, cols_to_drop):
    """
    Drops specified columns from a DataFrame, if they exist.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols_to_drop (list): List of column names to drop.

    Returns:
        pd.DataFrame: DataFrame with specified columns removed.
    """
    return df.drop(columns=[col for col in cols_to_drop if col in df.columns])


def check_missing_and_infinite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks for missing (NaN) and infinite values in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A summary DataFrame with counts of missing and infinite values per column.
    """

    numeric_df = df.select_dtypes(include=[np.number])
    
    missing_counts = df.isna().sum()
    inf_counts = np.isinf(numeric_df).sum()
    
    summary = pd.DataFrame({
        "missing": missing_counts,
        "infinite": inf_counts.reindex(df.columns, fill_value=0)  # match all columns
    })

    return summary


def plot_distribution(data: pd.DataFrame, column: str, bins: int = 100, 
                      title: str = None, xlim: tuple = None, log_scale: bool = False, save_path: str = None):
    """
    Plots the distribution of a specified numeric column in the dataframe.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): Name of the numeric column to plot.
        bins (int): Number of histogram bins.
        title (str): Custom title for the plot.
        xlim (tuple): Optional (min, max) limits for x-axis.
        log_scale (bool): Whether to use log scale on x-axis.
        save_path (str): If provided, saves the figure to this path.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    plt.figure(figsize=(10, 5))
    sns.histplot(data[column], bins=bins, kde=True, log_scale=log_scale)
    plt.title(title or f"{column} Distribution")
    plt.xlabel(column)
    if xlim:
        plt.xlim(xlim)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()
    
    
    
def analyze_numeric_column_over_time(df: pd.DataFrame, column: str, date_col: str = "date", sample_size: int = 5000):
    """
    Visualize and summarize a numeric column over time with scatter plots, boxplots, and key stats.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Numeric column to analyze.
        date_col (str): Name of the date column (default "date").
        sample_size (int): Number of samples for scatter plot if dataset is large.

    Returns:
        None. Shows plots and prints summary.
    """
    if column not in df.columns or date_col not in df.columns:
        raise ValueError(f"Column '{column}' or date column '{date_col}' not found in DataFrame.")

   
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')


    plot_data = df.sample(sample_size) if len(df) > sample_size else df

    plt.figure(figsize=(14, 6))
    sns.scatterplot(x=date_col, y=column, data=plot_data)
    plt.title(f'{column.capitalize()} Over Time (Sampled)')
    plt.xlabel('Date')
    plt.ylabel(column.capitalize())
    plt.show()

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    sns.boxplot(x=df[column], color='skyblue')
    plt.title(f'Complete {column.capitalize()} Distribution\n(Full Range)')
    plt.xlabel(f'{column.capitalize()}')
    plt.xticks(rotation=45)

    plt.axvline(x=0, color='red', linestyle=':', label='Zero Value')
    median_val = df[column].median()
    plt.axvline(x=median_val, color='green', linestyle='--', label=f'Median: {median_val:,.0f}')
    plt.legend()

    plt.subplot(1, 2, 2)
    p99 = df[column].quantile(0.99)
    detail_data = df[df[column] <= p99]
    sns.boxplot(x=detail_data[column], color='lightgreen')
    plt.title(f'Detailed View (0-99th Percentile)\n{p99:,.0f} cutoff')
    plt.xlabel(f'{column.capitalize()}')

    plt.tight_layout()
    plt.show()

    print(f"""
{column.capitalize()} Statistics:
Min: {df[column].min():,.0f}
Max: {df[column].max():,.0f}
Median: {median_val:,.0f}
Mean: {df[column].mean():,.0f}
IQR: {df[column].quantile(0.25):,.0f} to {df[column].quantile(0.75):,.0f}
""")
    
def drop_negative_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drops rows where any of the specified columns have negative values.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (list): List of column names to check for negatives.

    Returns:
        pd.DataFrame: Filtered dataframe with no negative values in the specified columns.
    """
    condition = (df[columns] >= 0).all(axis=1)
    return df[condition].copy()


def create_date_features(df):
 
    df['date'] = pd.to_datetime(df['date'])

   
    df['month'] = df['date'].dt.month.astype('int8')             
    df['year'] = df['date'].dt.year.astype('int16')
    df['quarter'] = df['date'].dt.quarter.astype('int8')       
    
   
    df['is_month_start'] = df['date'].dt.is_month_start.astype('int8')
    df['is_month_end'] = df['date'].dt.is_month_end.astype('int8')

    df['season'] = np.where(df['month'].isin([12, 1, 2]), 0, 1)
    df['season'] = np.where(df['month'].isin([3, 4, 5]), 1, df['season'])
    df['season'] = np.where(df['month'].isin([6, 7, 8]), 2, df['season'])
    df['season'] = np.where(df['month'].isin([9, 10, 11]), 3, df['season'])
    df['season'] = df['season'].astype('int8')
    
    return df


def time_based_train_val_split(X, y, val_ratio=0.2):
    n = len(X)
    val_size = int(n * val_ratio)
    train_size = n - val_size

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_val = X.iloc[train_size:]
    y_val = y.iloc[train_size:]

    return X_train, y_train, X_val, y_val


def aggregate_monthly_data(df, date_col='date', group_cols=['shop_id', 'item_id'], agg_dict=None):
    """
    Aggregates data to monthly totals per shop-item pair with flexible aggregation.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing date and features to aggregate.
    - date_col (str): Name of the datetime column.
    - group_cols (list): Columns to group by (default ['shop_id', 'item_id']).
    - agg_dict (dict): Dict of columns to aggregation functions, e.g.
        {'amount': 'sum', 'price': 'mean'}

    Returns:
    - pd.DataFrame: Monthly aggregated data.
    """
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')

    if agg_dict is None:
        raise ValueError("Please provide agg_dict with columns and aggregation functions.")

    df_monthly = (
        df.groupby(group_cols + [pd.Grouper(key=date_col, freq='M')])
        .agg(agg_dict)
        .reset_index()
    )

    return df_monthly

def create_lstm_sequences(df, window_size=12, feature_cols=['amount', 'price', 'sales'], target_col='sales'):
    """
    Create input sequences and targets for LSTM model.

    Parameters:
    - df: DataFrame sorted by ['shop_id_enc', 'item_id_enc', 'date']
    - window_size: Number of past months to use as input sequence (default 12)
    - feature_cols: List of column names to use as features per timestep
    - target_col: Column name for target value (next month sales)

    Returns:
    - X: np.array of shape (num_samples, window_size, num_features)
    - y: np.array of shape (num_samples,)
    """

    X_list = []
    y_list = []

    grouped = df.groupby(['shop_id_enc', 'item_id_enc'])

    for _, group in grouped:
        group = group.sort_values('date')
        features = group[feature_cols].values
        targets = group[target_col].values

     
        for i in range(len(group) - window_size):
            X_seq = features[i:i+window_size]
            y_target = targets[i+window_size]  
            X_list.append(X_seq)
            y_list.append(y_target)

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y
