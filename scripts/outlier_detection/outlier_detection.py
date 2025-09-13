import os
import sys
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def flag_outliers_iqr(df, columns):
    """Flag outliers using IQR method, with data type validation."""
    for column in columns:
        if column not in df.columns:
            logger.warning(f"Column '{column}' not in DataFrame.")
            continue
        
        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(df[column]):
            logger.error(f"Column '{column}' is not numeric (dtype: {df[column].dtype}). Skipping outlier detection.")
            logger.info(f"Sample values in '{column}': {df[column].head().tolist()}")
            continue
        
        # Convert to numeric if possible, coerce errors to NaN
        df[column] = pd.to_numeric(df[column], errors='coerce')
        
        # Check for sufficient numeric data
        numeric_count = df[column].count()
        if numeric_count < 4:  # Need at least 4 values for quartile calculation
            logger.warning(f"Column '{column}' has insufficient numeric data ({numeric_count} values). Skipping.")
            continue
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        logger.info(f"{column} → Q1: {Q1}, Q3: {Q3}, IQR: {IQR}, Lower: {lower}, Upper: {upper}")
        
        # Flag outliers
        outlier_mask = (df[column] < lower) | (df[column] > upper)
        df[f"{column}_outlier"] = outlier_mask.astype(int)
        
        # Log outlier statistics
        outlier_count = outlier_mask.sum()
        logger.info(f"Found {outlier_count} outliers in '{column}' ({outlier_count/len(df)*100:.1f}%)")
    
    return df

def main():
    # Read parameters from environment variables
    file_path = os.getenv('file_path', '')
    columns_str = os.getenv('columns', '')
    
    if not file_path or not columns_str:
        logger.error("Missing required parameters: file_path and columns")
        sys.exit(1)

    columns = [col.strip() for col in columns_str.split(",")]

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data from {file_path}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Column data types:\n{df.dtypes}")
        
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        sys.exit(1)

    # Process outliers
    df_out = flag_outliers_iqr(df, columns)
    
    # Generate output path
    output_path = file_path.replace('.csv', '_with_outliers.csv')
    df_out.to_csv(output_path, index=False)
    logger.info(f"Saved output to: {output_path}")

    # Print output path for backend to capture
    print(output_path)

if __name__ == '__main__':
    main()
