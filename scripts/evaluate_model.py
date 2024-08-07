import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('../data/BostonHousing.csv')

# Iterate over each column
for target_col in data.columns:
    # Define features (X) and target (y)
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Split the data into Training and Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and Train the Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    # Calculate Residuals
    residuals = y_test - y_pred
    
    # Plot the Prediction and Residuals
    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Predicted Values for Target: {target_col}')
    plt.show()
