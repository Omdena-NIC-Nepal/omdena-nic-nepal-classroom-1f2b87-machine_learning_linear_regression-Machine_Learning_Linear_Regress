import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
file_path='data/boston_housing.csv'
def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Handle missing values (if any)
    df = df.dropna()
    
    # Identify and handle outliers
    for col in df.columns:
        if df[col].dtype != 'object':
            q_low = df[col].quantile(0.01)
            q_high = df[col].quantile(0.99)
            df = df[(df[col] >= q_low) & (df[col] <= q_high)]
    
    # Encode categorical variables (if any)
    df = pd.get_dummies(df, columns=['chas'], drop_first=True)
    
    # Normalize/standardize numerical features
    scaler = StandardScaler()
    numerical_features = df.columns[df.dtypes != 'object']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Split the data into training and testing sets
    X = df.drop('medv', axis=1)
    y = df['medv']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Example usage
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data(file_path)
    X_train.to_csv('data/x_train1.csv', index=False)
    X_test.to_csv('data/x_test1.csv', index=False)
    y_train.to_csv('data/y_train1.csv', index=False)
    y_test.to_csv('data/y_test1.csv', index=False)
