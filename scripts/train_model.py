{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "07e19e5d-ccf8-4029-8b21-744ffc9c0701",
   "metadata": {},
   "outputs": [],
   "source": [
    "# scripts/train_model.py\n",
    "import pandas as pd\n",
    "from sklearn.linear_model import LinearRegression\n",
    "import joblib\n",
    "\n",
    "def train_model(train_X_file, train_y_file, model_output_file):\n",
    "    X_train = pd.read_csv(train_X_file)\n",
    "    y_train = pd.read_csv(train_y_file)\n",
    "    model = LinearRegression()\n",
    "    model.fit(X_train, y_train)\n",
    "    joblib.dump(model, model_output_file)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    train_model('../data/X_train.csv', '../data/y_train.csv', '../models/linear_regression_model.pkl')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "41e42fd4-ae86-42f9-8cc0-ab7a5f3ad0c4",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
