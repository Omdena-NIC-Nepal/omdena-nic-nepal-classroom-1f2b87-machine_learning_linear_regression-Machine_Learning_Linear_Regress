import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

def main():
    # Load the data
    data = pd.read_csv('../data/BostonHousing.csv')
    print(data.head())

    # Separate features and target
    X = data.drop(columns=['medv'])
    y = data['medv']

    print(X.head())
    print(y.head())

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape)

    # Standard Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate Linear Regression Model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # Plot the results
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Linear Regression Prediction')
    plt.show()

    # Classification using Random Forest Classifier
    # Binning the target variable 'medv'
    bins = [0, 20, 35, 50]
    labels = [0, 1, 2]
    y_train_class = pd.cut(y_train, bins=bins, labels=labels, include_lowest=True)
    y_test_class = pd.cut(y_test, bins=bins, labels=labels, include_lowest=True)

    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train_class)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_:.4f}")

if __name__ == "__main__":
    main()
