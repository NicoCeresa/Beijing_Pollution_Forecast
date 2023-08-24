import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential, load_model

class Proj4: 

    def rmse(actual, preds):
        """
        inputs
        actual: known numbers
        preds: predicted values

        outputs
        Root mean squared esrror of the data
        """
        model_rmse = np.sqrt(mse(actual, preds))
        return model_rmse

    def wind_encode(s):
        """
        inputs
        s: Series containing "SE, NE, NW, SW"
        
        outputs
        encoded version of the inputted Series
        """
        if s == "SE":
            return 1
        elif s == "NE":
            return 2
        elif s == "NW":
            return 3
        else:
            return 4

    def scale_data(X_data):
        scale = StandardScaler()
        return scale.fit_transform(X_data)

    def df_to_X_y(df, window_size):
        """
        inputs
        df: Dataframe
        window_size: amount to move by 
        
        outputs
        array of features and array of target
        """
        df_as_np = df.to_numpy()
        X = []
        y = []
        for i in range(len(df_as_np)-window_size):
            row = [[a] for a in df_as_np[i:i+window_size]]
            X.append(row)
            label = df_as_np[i+window_size]
            y.append(label)
        return np.array(X), np.array(y)

    def load_model():
        """
        inputs:
        None
        
        outputs:
        Data divided into training, validation, and testing data
        """
        WINDOW_SIZE=5

        train = pd.read_csv('./data/LSTM-Multivariate_pollution.csv')
        test = pd.read_csv('./data/pollution_test_data1.csv')
        
        train.index = pd.to_datetime(train['date'], format='%Y.%m.%d %H:%M:%S')
        poll = train['pollution']
        train["wind_dir"] = train["wnd_dir"].apply(Proj4.wind_encode)
        train = train.drop(["wnd_dir", 'date'], axis=1)
        

        X, y = Proj4.df_to_X_y(poll, WINDOW_SIZE)
        X_train, y_train = X[:35000], y[:35000]
        X_train = StandardScaler().fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val, y_val = X[35000:], y[35000:65000]
        X_val = StandardScaler().fit_transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)


        target2 = test['pollution']
        test["wind_dir"] = test["wnd_dir"].apply(Proj4.wind_encode)
        test = test.drop(["wnd_dir"], axis=1)
        X_test, y_test = Proj4.df_to_X_y(target2, WINDOW_SIZE)
        X_test = StandardScaler().fit_transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        return X_train, y_train, X_val, y_val, X_test, y_test
    

    def model():
        """
        inputs:
        None
        
        outputs:
        LSTM Recurrent Neural Network
        """
        model1 = Sequential()
        model1.add(InputLayer((5,1)))
        model1.add(LSTM(64))
        model1.add(Dense(8, 'relu'))
        model1.add(Dense(1, 'linear'))

        return model1

        
    def predict(X, actual, label):
        """
        inputs:
        X - features
        actual - known values to compare against
        label - Train, Validation, or Test
        
        outputs
        Dataframe of predictions and actual data
        """
        model1 = load_model('model1/')
        predictions = model1.predict(X).flatten()
        results = pd.DataFrame(data={f'{label} Predictions': predictions,
                                        'Actual':actual})
        return results

    def plot(results, actual, label):
        """
        inputs:
        results: Dataframe outputted by predict
        actual - known values to compare against
        label - Train, Validation, or Test
        
        outputs
        plots of actual vs predicted with the RMSE as the title
        """
        plt.plot(results[f'Actual'][:100], label='Actual')
        plt.plot(results[f'{label} Predictions'][:100], label='Predictions', c='r')
        preds = results[f'{label} Predictions']
        plt.title(f"{label} RMSE: {Proj4.rmse(actual, preds)}")
        plt.legend()
        return plt.show()

class Run:

    def run(self):

        X_train, y_train, X_val, y_val, X_test, y_test = Proj4.load_model()
        
        model = Proj4.model()
        cp = ModelCheckpoint('model1/', save_best_only=True)
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
        model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=20, verbose=1, callbacks=[cp])

        print('Predicting Training Data:')
        print()

        train_preds = Proj4.predict(X_train, y_train, 'Train')

        Proj4.plot(train_preds, y_train, 'Train')
#------------------------------------------------------------
        print('Predicting Validation Data:')
        print()

        val_preds = Proj4.predict(X_val, y_val, 'Validation')

        Proj4.plot(val_preds, y_val, 'Validation')
#------------------------------------------------------------
        print('Predicting Testing Data:')
        print()

        test_preds = Proj4.predict(X_test, y_test, 'Test')

        Proj4.plot(test_preds, y_test, 'Test')


if __name__ == "__main__":
    exp = Run()
    exp.run()