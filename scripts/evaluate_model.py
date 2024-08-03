import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

def evaluate_model(x_test,y_test,model):
    y_pred=model.predict(x_test)
    
    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)

    return mse,r2,y_pred

    # Example usage
if __name__ == "__main__":
    X_test = pd.read_csv('data/X_test1.csv')
    y_test = pd.read_csv('data/y_test1.csv')
    
    model = joblib.load('linear_regression_model1.pkl')
    
    mse, r2, y_pred = evaluate_model(X_test, y_test, model)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    
    # Plot residuals
    residuals = y_test.values.flatten() - y_pred.flatten()
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()
    
    # Plot predicted vs actual values
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.show()
