import pandas as pd
import numpy as np
from scipy.stats import shapiro

def mean_imputation(df, group_col=None, num_cols=None, method='mean', add_indicators=True):
    """
    Performs mean/median imputation on numeric columns, optionally grouped by a categorical column.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing data to be imputed
    group_col : str, optional
        Column name to group by for imputation (if None, global imputation is used)
    num_cols : list, optional
        List of numeric column names to impute (if None, all numeric columns are used)
    method : str, optional
        Imputation method: 'mean', 'median', or 'auto' (uses normality test to decide)
    add_indicators : bool, optional
        Whether to add binary indicator columns for missing values

    Returns:
    --------
    tuple: (df_imputed, impute_dict)
        df_imputed: DataFrame with imputed values
        impute_dict: Dictionary with imputation values used for each column/group
    """

    # Create a copy of the dataframe to avoid modifying the original
    df_imputed = df.copy()

    # If num_cols not specified, use all numeric columns
    if num_cols is None:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Initialize the imputation dictionary
    impute_dict = {col: {} for col in num_cols}

    # Define groups for imputation
    if group_col is None:
        # If no group column, use a dummy group
        groups = [('all', df_imputed)]
    else:
        # Get unique groups and create (group_value, group_df) pairs
        unique_groups = df_imputed[group_col].unique()
        groups = [(group, df_imputed[df_imputed[group_col] == group]) for group in unique_groups]

    # Perform imputation for each column and group
    for col in num_cols:
        # Add missing indicator if requested
        if add_indicators:
            indicator_name = f"{col}_ismissing"
            df_imputed[indicator_name] = df_imputed[col].isna()

        for group_val, group_df in groups:
            # Calculate statistics for the group
            col_data = group_df[col].dropna()

            # Skip if no data available for this group
            if len(col_data) == 0:
                continue

            col_mean = col_data.mean()
            col_median = col_data.median()

            # Determine imputation value based on method
            if method == 'mean':
                impute_value = col_mean
            elif method == 'median':
                impute_value = col_median
            elif method == 'auto':
                # Use normality test to decide between mean and median
                if len(col_data) >= 3:  # Shapiro-Wilk requires at least 3 observations
                    _, p_value = shapiro(col_data)
                    # If normal (p > 0.05), use mean; otherwise use median
                    impute_value = col_mean if p_value > 0.05 else col_median
                else:
                    # Not enough data for normality test, default to median
                    impute_value = col_median
            else:
                raise ValueError(f"Invalid method: {method}. Use 'mean', 'median', or 'auto'.")

            # Store imputation value in dictionary
            impute_dict[col][group_val] = impute_value

            # Apply imputation
            if group_col is None:
                # Global imputation
                df_imputed[col].fillna(impute_value, inplace=True)
            else:
                # Group-specific imputation
                mask = (df_imputed[group_col] == group_val) & (df_imputed[col].isna())
                df_imputed.loc[mask, col] = impute_value

    return df_imputed, impute_dict
