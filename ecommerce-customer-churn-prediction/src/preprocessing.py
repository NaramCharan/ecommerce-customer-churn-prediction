
import pandas as pd
from sklearn.model_selection import train_test_split

class data_separation:
    def __init__(self):
        self.df = pd.read_excel('../dataset/E_Commerce_Dataset.xlsx', sheet_name='E Comm')


    def spliting(self):
        X = self.df.drop(columns=['Churn', 'CustomerID'])
        Y = self.df['Churn']
        X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=34)
        return X_train, x_test, Y_train, y_test





