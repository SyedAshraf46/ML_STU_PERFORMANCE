
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class STUDENT_PERFORMANCE_PREDICTION:

    def __init__(self, path):
        try:
            # Read dataset
            self.df = pd.read_csv(path)

            print("Dataset Loaded Successfully")
            print(self.df.head())

            # Separate independent and dependent variables
            self.X = self.df.iloc[:, :-1]
            self.Y = self.df.iloc[:, -1]

            # Convert categorical columns into dummy variables
            self.X = pd.get_dummies( self.X, drop_first=True, dtype=int)

            # Split training and testing data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.Y,
                test_size=0.2,
                random_state=42)

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            print(f"Error in line no: {er_line.tb_lineno} : {er_msg} : {er_ty}")

    # Training data
    def train_data(self):
        try:
            print("\nTraining Data")

            self.reg = LinearRegression()

            self.reg.fit(self.X_train, self.y_train)

            # Coefficient and intercept
            a = self.reg.coef_
            b = self.reg.intercept_

            print(f"m_value: {a}")
            print(f"c_value: {b}")

            # Training prediction
            train_pred = self.reg.predict(self.X_train)

            # R2 Score
            y_mean = self.y_train.mean()

            numerator = ((self.y_train - train_pred) ** 2).sum()
            denominator = ((self.y_train - y_mean) ** 2).sum()

            r2 = 1 - (numerator / denominator)

            print(f"Training R2 Score: {r2}")

            # Mean Squared Error
            mse = ((self.y_train - train_pred) ** 2).mean()

            print(f"Training MSE: {mse}")

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            print(f"Error in line no: {er_line.tb_lineno} : {er_msg} : {er_ty}")

    # Testing data
    def test_data(self):
        try:
            print("\nTesting Data")

            test_pred = self.reg.predict(self.X_test)

            # R2 Score
            y_test_mean = self.y_test.mean()

            numerator = ((self.y_test - test_pred) ** 2).sum()
            denominator = ((self.y_test - y_test_mean) ** 2).sum()

            r2 = 1 - (numerator / denominator)

            print(f"Testing R2 Score: {r2}")

            # Mean Squared Error
            mse = ((self.y_test - test_pred) ** 2).mean()

            print(f"Testing MSE: {mse}")

            # Save model
            with open("Std.Pred.pkl", "wb") as f:
                pickle.dump(self.reg, f)

            print("Model saved successfully")

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            print(f"Error in line no: {er_line.tb_lineno} : {er_msg} : {er_ty}")


# Main execution
if __name__ == "__main__":

    obj = STUDENT_PERFORMANCE_PREDICTION("Student_Performance.csv")

    obj.train_data()

    obj.test_data()
