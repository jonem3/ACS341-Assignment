"""
Author: Matthew Philip Jones
Registration Number: 200326702
Email: mjones18@sheffield.ac.uk
Date Created: 12th March 2024

This script is my submission for ACS341 - Machine Learning Assignment 2
I have chosen to use Python as in my experience data processing is a lot more streamlined than in MATLAB
I will be specifying types in all function declarations for the benefit of the reader, please note this is not
required as standard in Python
"""
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import keras_tuner as kt
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from colorama import Fore
from sklearn.cluster import DBSCAN

# Weather type categories for One-Hot Encoding
WEATHER_COLUMN_TYPES = [
    'clear-night',
    'clear-day',
    'partly-cloudy-night',
    'partly-cloudy-day',
    'cloudy',
    'fog',
    'wind',
    'rain',
    'snow'
]

# Columns that I have deemed irrelevant to the topic we are trying to predict
COLUMNS_TO_DROP = [
    'RadonLevel_Bqm3',
    'windBearing',
    'precipProbability',
]


def find_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds outliers in each column using Z-Score, outliers are then replaced with the edge value for the direction of
    the outlier.
    :param df: The data frame to be cleaned
    :return: The cleaned data frame
    """
    for col in df.columns.tolist():
        data = df[col].tolist()
        mean = df[col].mean()

        standard_deviation = df[col].std()
        for data_point in data:
            zscore = (data_point - mean) / standard_deviation
            if zscore > 4:
                data[data.index(data_point)] = mean + (standard_deviation * 4)
            elif zscore < -4:
                data[data.index(data_point)] = mean - (standard_deviation * 4)

        df[col] = data
    return df


def build_scalars(df: pd.DataFrame) -> list:
    """
    Calculates the max & min values of each column in the data frame and stores them for scaling / descaling later
    :param df: the data frame containing the columns to be scaled
    :return: array of scalars for each column
    """
    columns = df.columns.tolist()
    data_scaler = []
    for col in columns:
        max_col = df[col].max()  # Finds maximum value in column
        min_col = df[col].min()  # Finds minimum value in column
        data_scaler.append([col, max_col, min_col])
    return data_scaler


def remove_collinearity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes collinearity between two columns using Pearson correlation coefficient by checking every combination of
    columns, excluding the output column, as we want to keep columns which correlate with the output.
    If correlation is above 0.9 then the second column is removed
    :param df: data frame to be checked for collinearity
    :return: cleaned data frame
    """
    # TODO - Basing initial collinear detection on: https://www.stratascratch.com/blog/a-beginner-s-guide-to-collinearity-what-it-is-and-how-it-affects-our-regression-model/

    columns = df.columns.tolist()[1:]
    for col1 in columns:
        X = df[col1].to_numpy()
        for col2 in columns:
            if col1 != col2:
                Y = df[col2].to_numpy()
                pearson = np.corrcoef(X, Y)[0, 1]
                if abs(pearson) > 0.8:
                    print(Fore.RED + "Dropped:", col2, "Due to collinearity of", pearson, "with:", col1 + Fore.RESET)

                    plt.plot(df[col1], df[col2], 'o')
                    plt.title('Collinearity Report - (' + col1 + ', ' + col2 + ')')
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    plt.grid(True)
                    plt.show()

                    columns.remove(col2)
                    df.drop(col2, axis=1, inplace=True)
    return df


def data_processor(df: pd.DataFrame, data_scaler: list) -> tuple:
    """
    Main function for processing the data frame and scaling the data frame
    Cleans out irrelevant columns before calling other data processing functions (e.g. collinearity, outliers, etc.)
    :param df: Data Frame to be processed for model training
    :param data_scaler: Scalar values for data scaling so that data can be descaled after model training / prediction
    :return: Data Frame after processing and data scalars
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replaces inf values with NaN values
    df.dropna(inplace=True, axis=0, how='any')  # Drops all NaN values

    df = pd.get_dummies(df, columns=['WeatherIcon'], prefix='', prefix_sep='')  # One-Hot Encoding on categorical data

    for col in WEATHER_COLUMN_TYPES:
        df[col] = df[col].map({True: 1, False: 0})  # Mapping True & False from categorical to 1 & 0

    # for col in df.columns.tolist():
    #     if 'KW' not in col.upper():
    #         warning = "Column '" + col + "' has no kW value"
    #         print(Fore.RED + warning + Fore.RESET)
    #         df.drop(col, axis=1, inplace=True)  # Dropping all columns that do not involve kW usage / generation

    df = remove_collinearity(df)
    df = find_outliers(df)

    columns = df.columns.tolist()  # Obtain a list of remaining column names

    if len(data_scaler) != len(columns):
        data_scaler = build_scalars(df)  # Build scalars array for data scaling

    for col in columns:
        df[col] = (df[col] - data_scaler[columns.index(col)][2]) / (
                data_scaler[columns.index(col)][1] - data_scaler[columns.index(col)][2])  # Scaling data for training

    return df, data_scaler


def linear_regression(df: pd.DataFrame) -> None:
    """
    Linear regression model generation & testing
    :param df: Data Frame containing data to build OLS Regression model
    """
    x, y, x_test, y_test = split_dataset(df)  # Split data into training & testing

    model = sm.OLS(y, x)  # Generate Ordinary Least Squares Linear Regression model

    results = model.fit()  # Run OLS to get coefficients

    print(results.summary())

    prediction = results.predict(x_test)  # Generate predicted data from unseen testing data

    quality_graphs(prediction, y_test, 'Linear Regression')


def split_dataset(df: pd.DataFrame) -> tuple:
    """
    Splits data frame into training and test sets
    :param df: Full data set to be split
    :return: Training Input Set, Training Output Set, Testing Input Set, Testing Output Set
    """
    train_dataset = df.sample(frac=0.8, random_state=0, axis=0)  # Sample data for training (80%)
    test_dataset = df.drop(train_dataset.index)  # Drop training data to leave testing data

    # Split datasets into Inputs / Outputs
    x = train_dataset.iloc[:, 1:]
    x_test = test_dataset.iloc[:, 1:]

    y = train_dataset.iloc[:, 0]
    y_test = test_dataset.iloc[:, 0]

    # Add constant 1 column to inputs for offset compensation
    x = sm.add_constant(x)
    x_test = sm.add_constant(x_test)

    return x, y, x_test, y_test


def build_model(hp) -> tf.keras.models.Model:
    """
    Builds model using Keras API from Tensorflow

    :param hp: Model hyperparameters
    :return: Tensorflow Model
    """
    hp_units = hp.Int('units', min_value=1, max_value=100, step=10)  # Define first layer units possibilities for train

    model = keras.Sequential([
        keras.layers.Dense(units=hp_units, activation='sigmoid'),
        keras.layers.Dense(64, activation='sigmoid'),
        keras.layers.Dense(1)
    ]
    )

    hp_learning_rate = hp.Choice('learning_rate',
                                 values=[0.001, 0.01, 0.1, 1.0])  # Possibilities for learning rate

    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=['accuracy'])

    return model


def plot_loss(history) -> None:
    loss = (data_scaler[0][1] - data_scaler[0][2]) * (history.history['loss'] + data_scaler[0][2])
    val_loss = (data_scaler[0][1] - data_scaler[0][2]) * (history.history['val_loss'] + data_scaler[0][2])
    plt.plot(loss, label='Loss (MSE)')
    plt.plot(val_loss, label='Validation Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Error [kW^2]')
    plt.legend()
    plt.grid(True)
    plt.show()


def deep_neural_network(df: pd.DataFrame) -> None:
    train_features, train_labels, test_features, test_labels = split_dataset(df)

    tuner = kt.Hyperband(build_model,
                         objective='val_loss',
                         max_epochs=50,
                         factor=3,
                         directory='./checkpoints',
                         project_name='acs341_assignment'
                         )

    callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    tuner.search(train_features, train_labels, epochs=50, validation_split=0.3, callbacks=[callback])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps)

    dnn_household_energy_model = tuner.hypermodel.build(best_hps)

    history = dnn_household_energy_model.fit(
        train_features,
        train_labels,
        validation_split=0.3,
        verbose=1,
        epochs=50,
        callbacks=[callback]
    )

    print(dnn_household_energy_model.summary())

    plot_loss(history)
    dnn_household_energy_model.evaluate(test_features, test_labels, verbose=1)
    prediction = dnn_household_energy_model.predict(test_features).flatten()

    quality_graphs(prediction, test_labels, 'Deep Neural Network')


def quality_graphs(prediction: list, test_labels: list, title: str) -> None:
    prediction = (data_scaler[0][1] - data_scaler[0][2]) * (prediction + data_scaler[0][2])
    test_labels = (data_scaler[0][1] - data_scaler[0][2]) * (test_labels + data_scaler[0][2])

    print(len(prediction), len(test_labels))

    plt.plot(prediction, test_labels, 'o')
    plt.xlabel('Prediction Energy Requested From Grid [kW]')
    plt.ylabel('Actual Energy Requested From Grid [kW]')
    plt.title(title + ' Prediction vs Actual Energy Requested From Grid')
    plt.grid()
    plt.show()

    plt.plot(test_labels.index, test_labels, 'o')
    plt.plot(test_labels.index, prediction, 'o')
    plt.xlabel('Index')
    plt.ylabel('Energy Requested From Grid')
    plt.legend(['Actual Energy Requested From Grid', 'Prediction'])
    plt.title(title + ' Prediction vs Actual Energy Requested From Grid Indexed')
    plt.grid(True)
    plt.show()

    error = prediction - test_labels
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [kW]')
    plt.ylabel('Count')
    plt.title(title + ' Prediction Error')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('assignment_docs/household_energy_data.csv')
    df = df.drop(columns=COLUMNS_TO_DROP)
    data_scaler = []
    df, data_scaler = data_processor(df, data_scaler)
    linear_regression(df)
    deep_neural_network(df)
