{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9690807f-893b-4166-9ae2-b0d01db68c19",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "8091f8ff-af58-49f4-b3c0-b5f9e60eac9a",
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_data(input_file, output_train_X, output_test_X, output_train_y, output_test_y):\n",
    "    df = pd.read_csv(input_file)\n",
    "    df.fillna(df.mean(), inplace=True)\n",
    "    df = pd.get_dummies(df, columns=['CHAS'], drop_first=True)\n",
    "    scaler = StandardScaler()\n",
    "    scaled_features = scaler.fit_transform(df.drop('MEDV', axis=1))\n",
    "    scaled_df = pd.DataFrame(scaled_features, columns=df.columns[:-1])\n",
    "    X = scaled_df\n",
    "    y = df['MEDV']\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "    X_train.to_csv(output_train_X, index=False)\n",
    "    X_test.to_csv(output_test_X, index=False)\n",
    "    y_train.to_csv(output_train_y, index=False)\n",
    "    y_test.to_csv(output_test_y, index=False)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    preprocess_data('../data/HousingData.csv', '../data/X_train.csv', '../data/X_test.csv', '../data/y_train.csv', '../data/y_test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "67da9c52-7f5e-4812-822c-486f0eca0fd7",
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
