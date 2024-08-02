# data_preprocessing.py

# Importing necessary libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data(file_path):
    """
    Load the dataset from a specified file path.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

def handle_missing_values(df):
    """
    Handle missing values by filling them with the median of each column.
    
    Parameters:
    df (DataFrame): The input dataframe.
    
    Returns:
    DataFrame: Dataframe with missing values handled.
    """
    return df.fillna(df.median())

def handle_outliers(df, threshold=3):
    """
    Handle outliers by removing rows with Z-scores above a certain threshold.
    
    Parameters:
    df (DataFrame): The input dataframe.
    threshold (float): The Z-score threshold to identify outliers.
    
    Returns:
    DataFrame: Dataframe with outliers removed.
    """
    z_scores = np.abs(stats.zscore(df))
    return df[(z_scores < threshold).all(axis=1)]

def encode_categorical(df):
    """
    Encode categorical variables using one-hot encoding.
    
    Parameters:
    df (DataFrame): The input dataframe.
    
    Returns:
    DataFrame: Dataframe with categorical variables encoded.
    """
    return pd.get_dummies(df, columns=['chas'], drop_first=True)

def normalize_features(df):
    """
    Normalize/standardize numerical features.
    
    Parameters:
    df (DataFrame): The input dataframe.
    
    Returns:
    DataFrame: Dataframe with normalized features.
    """
    scaler = StandardScaler()
    numerical_cols = df.drop('medv', axis=1).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def split_data(df):
    """
    Split the data into training and testing sets.
    
    Parameters:
    df (DataFrame): The input dataframe.
    
    Returns:
    tuple: Training and testing sets (X_train, X_test, y_train, y_test).
    """
    X = df.drop('medv', axis=1)
    y = df['medv']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    # Load the data

    import os

    file_path = "data/BostonHousing.csv"

    if os.path.exists(file_path):
        print("File found.")
    else:
        print("File not found. Check the path.")

    df = load_data(file_path)
    
    # Display the first 10 rows of the dataset
    print("Initial Data:")
    print(df.head(10))
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Handle outliers
    df = handle_outliers(df)
    
    # Display column names
    print("\nColumns in the dataset:")
    print(df.columns)
    
    # Encode categorical variables
    df = encode_categorical(df)
    
    # Normalize features
    df = normalize_features(df)
    
    # Display the first few rows of the transformed dataset
    print("\nTransformed Data:")
    print(df.head())

    df.to_csv("data/preprocessed_data.csv", index=False)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Display the shapes of the training and testing sets
    print("\nTraining set shape:", X_train.shape, y_train.shape)
    print("Testing set shape:", X_test.shape, y_test.shape)

if __name__ == "__main__":
    main()
