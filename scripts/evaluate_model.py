import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import joblib
import argparse


def evaluate_linear_regression_model(data_folder, model_folder):
    """
    Evaluates each linear regression model using datasets from the given folder and saves the metrics.
    
    Parameters:
        data_folder (str): Path to the folder containing CSV files of the datasets.
        model_folder (str): Path to the folder containing trained models.
    """
    results = []
    for model_filename in os.listdir(model_folder):
        if model_filename.endswith('_model.pkl'):
            model_path = os.path.join(model_folder, model_filename)
            model_name = os.path.splitext(model_filename)[0]

            # Extract corresponding data filename
            data_filename = model_name.replace("_model", "") + ".csv"
            data_path = os.path.join(data_folder, data_filename)

            # Load the dataset
            data = pd.read_csv(data_path)

            # Define features (X) and target (y)
            X = data.drop(columns=["medv"])
            y = data["medv"]

            # Split the dataset into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            # Load the trained model
            model = joblib.load(model_path)

            # Make predictions on the data
            y_pred = model.predict(X_test)

            # Calculate evaluation metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            explained_variance = explained_variance_score(y_test, y_pred)


            results.append({
                "Model": model_name,
                "MSE":mse,
                "r2":r2,
                "rmse":rmse,
                "v_score":explained_variance
            })
    df_eval = pd.DataFrame(results)
    print(df_eval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate linear regression models.")
    parser.add_argument("data_folder", type=str,
                        help="Path to the folder containing CSV files of the datasets.")
    parser.add_argument("model_folder", type=str,
                        help="Path to the trained model file.")

    args = parser.parse_args()

    evaluate_linear_regression_model(
        args.data_folder, args.model_folder)
