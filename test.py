import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Plotting and Visualizing data

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
class regLinear:

    def liearReg():

        data = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv', usecols=["Global_Sales", "JP_Sales", "EU_Sales"])
        print(data.describe())
        x = data.iloc[:, 0:1].values
        y = data.iloc[:, 1]
        #Usuwanie pustych wartośći
        data=data.dropna()

        print("######5 pierwszych wierszy######################")
        print(data.head())
        print("################################################")
        print("To co sie duplikuje")
        duplicated = data[data.duplicated()]
        print(duplicated.shape)
        print("##########################")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
        print(x[:10])
        print('\n')
        print(y[:10])

        # Model Import and Build
        regressor = LinearRegression()
        regressor.fit(x_train, y_train)

        pred = regressor.predict(x_test)

        # Wykres
        ## Check the fitting on training set
        plt.scatter(x_train, y_train)
        plt.plot(x_train, regressor.predict(x_train), color='black')

        plt.title('Fit on training set')
        plt.xlabel('X-Train')
        plt.ylabel('Y-Train')
        plt.show()
        ###########################
        p2 = sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)
        X2_train = p2.fit_transform(x_train)
        X2_test = p2.fit_transform(x_test)
        regressor.fit(X2_train, y_train)
        y_train_pred = regressor.predict(X2_train)
        y_test_pred = regressor.predict(X2_test)
        mse = sklearn.metrics.mean_squared_error(y_test, y_test_pred)
        mae = sklearn.metrics.mean_absolute_error(y_test, y_test_pred)
        # okreslenie bledu
        print("określienie błedu")
        print("MSE")
        print(mse)
        print("MAE")
        print(mae)
        print(y_train_pred)
        print(y_test_pred)
        print(pred)
regresja=regLinear.liearReg()
