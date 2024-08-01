import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def train_model(x_train, y_train):
    model=LinearRegression()
    model.fit(x_train,y_train)
    return model

# Example usage
if __name__=="__main__":
    x_train=pd.read_csv('data/x_train1.csv')
    y_train=pd.read_csv('data/y_train1.csv')

    model=train_model(x_train,y_train)

    # Save the model
    joblib.dump(model,'linear_regression_model1.pkl')
