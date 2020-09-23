import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import sys

sys.path.append('src')
sys.path.append('src/visualization')

from visualization.visualize import *


def compile_model(n_features):
    model=Sequential()
    model.add(Dense(12, input_dim=n_features, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))


    #compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def fit_model(model, features, labels, n_epochs=10, n_batch=10, n_val_split=0.1):
    history = model.fit(features, labels, epochs=n_epochs, batch_size=n_batch, validation_split=n_val_split)
    return history

if __name__ == '__main__':

    # create untrained model
    model = compile_model(8)
    # load data
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    # train your model on the data
    history = fit_model(model, X_train, y_train, n_epochs=50, n_batch=30, n_val_split=0.2)
    loss_plot(history)
    model_path = 'models/stk880_model_v1.h5'
    history.model.save(model_path)

