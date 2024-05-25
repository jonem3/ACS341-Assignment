"""
Author: Matthew Philip Jones
Email: mjones18@sheffield.ac.uk
Date Created: 12th March 2024

This script is my submission for ACS341 - Machine Learning Assignment 2
I have chosen to use Python as in my experience data processing is a lot more streamlined than in MATLAB
I will be specifying types in all function declarations for the benefit of the reader, please note this is not
required as standard in Python

All used libraries have been stored to a requirements file for your convenience, full setup instructions can be found
in README.md in the root folder of this project
"""
import os

import keras_tuner as kt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
import statsmodels.api as sm
import tensorflow as tf
from colorama import Fore
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from tensorflow import keras

"""
------------------ SETTINGS ------------------
"""

CSV_FILE_PATH = 'assignment_docs/household_energy_data.csv'

"""
------------------ TASK 1 ------------------
"""

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


def data_processor(df_processing: pd.DataFrame, data_scale: list) -> tuple:
    """
    Main function for processing the data frame and scaling the data frame
    Cleans out irrelevant columns before calling other data processing functions (e.g. collinearity, outliers, etc.)
    :param df_processing: Data Frame to be processed for model training
    :param data_scale: Scalar values for data scaling so that data can be descaled after model training / prediction
    :return: Data Frame after processing and data scalars
    """

    df_processing = df_processing.drop(columns=COLUMNS_TO_DROP)

    df_processing.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan,
        inplace=True
    )  # Replaces inf values with NaN values

    df_processing.dropna(
        how='any',
        axis=0,
        inplace=True
    )  # Drops all NaN values

    df_processing = pd.get_dummies(
        df_processing,
        columns=['WeatherIcon'],
        prefix='',
        prefix_sep=''
    )  # One-Hot Encoding on categorical data

    for col in WEATHER_COLUMN_TYPES:
        df_processing[col] = df_processing[col].map(
            {
                True: 1,
                False: 0
            }
        )  # Mapping True & False from categorical to 1 & 0

    df_processing = find_outliers(df_processing)

    if len(data_scale) + 1 != len(df_processing.columns.tolist()):
        data_scale = build_scalars(df_processing)  # Build scalars array for data scaling

    df_processing = min_max_scale(df_processing)

    df_processing = remove_collinearity(df_processing)

    sns.pairplot(df_processing.iloc[:, 0:5])  # Generate pair-plot of first 5 columns
    plt.show()

    df_processing = pca_reduce_dimension(df_processing)

    return df_processing, data_scale


def build_scalars(df_reference: pd.DataFrame) -> list:
    """
    Calculates the max & min values of each column in the data frame and stores them for scaling / descaling later
    :param df_reference: the data frame containing the columns to be scaled
    :return: array of scalars for each column
    """
    columns = df_reference.columns.tolist()
    data_scaler_generator = []
    for col in columns:
        if col != 'WeatherIcon':
            max_col = df_reference[col].max()  # Finds maximum value in column
            min_col = df_reference[col].min()  # Finds minimum value in column
            data_scaler_generator.append(
                [
                    col,
                    max_col,
                    min_col
                ]
            )
    return data_scaler_generator


def min_max_scale(df_to_scale: pd.DataFrame) -> pd.DataFrame:
    """
    Scales each column in the data frame to be within the range of [0, 1]
    :param df_to_scale: Unscaled data frame
    :return: Scaled data frame
    """
    for col in df_to_scale.columns.tolist():
        if col != 'WeatherIcon':
            df_to_scale[col] = (
                    (df_to_scale[col] - df_to_scale[col].min()) /
                    (df_to_scale[col].max() - df_to_scale[col].min())
            )
    return df_to_scale


def find_outliers(df_with_outliers: pd.DataFrame) -> pd.DataFrame:
    """
    Finds outliers in each column using Z-Score, outliers are then replaced with an interpolated value
    based on the other values in that row
    :param df_with_outliers: The data frame to be cleaned
    :return: The cleaned data frame
    """
    threshold = 3
    for col in df_with_outliers.columns.tolist():
        if col not in WEATHER_COLUMN_TYPES:
            z = np.abs(stats.zscore(df_with_outliers[col]))
            outliers = df_with_outliers[z > threshold].index
            df_with_outliers.loc[outliers, col] = np.nan
    df_with_outliers.interpolate(inplace=True, axis=0)
    return df_with_outliers


def remove_collinearity(df_collinearity_check: pd.DataFrame) -> pd.DataFrame:
    """
    Removes collinearity between two columns using Pearson correlation coefficient by checking every combination of
    columns, excluding the output column, as we want to keep columns which correlate with the output.
    If correlation is above 0.9 then the second column is removed
    :param df_collinearity_check: data frame to be checked for collinearity
    :return: cleaned data frame
    """

    correlation_matrix = df_collinearity_check.corr()
    top_corr_features = correlation_matrix.index

    sns.heatmap(
        df_collinearity_check[top_corr_features].corr(),
        annot=False,
        cmap="RdYlGn"
    )
    plt.show()

    columns = df_collinearity_check.columns.tolist()[1:]
    for col1 in columns:
        x = df_collinearity_check[col1].to_numpy()
        for col2 in columns:
            if col1 != col2:
                y = df_collinearity_check[col2].to_numpy()
                pearson = np.corrcoef(x, y)[0, 1]
                if abs(pearson) > 0.9:
                    print(f'{Fore.RED} Dropped {col2} Due to collinearity of: {pearson} with: {col1} {Fore.RESET}')

                    plt.plot(df_collinearity_check[col1], df_collinearity_check[col2], 'o')
                    plt.title(
                        f'Collinearity Report - ({col1}, {col2}: {str(round((100 * float(pearson)), 2))}%)')
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    plt.grid(True)
                    plt.show()

                    columns.remove(col2)
                    df_collinearity_check.drop(
                        col2,
                        axis=1,
                        inplace=True
                    )
    return df_collinearity_check


def pca_reduce_dimension(df_pre_pca: pd.DataFrame) -> pd.DataFrame:
    """
    Reduces the data frame using PCA to reduce dimensions, specifically targeting all the weather based data
    in the dataset
    :param df_pre_pca: Dataframe before dimensionality reduction
    :return: Reduced data frame
    """
    weather_cols = (col for col in df_pre_pca.columns.tolist() if "KW" not in col.upper())

    df_weather = df_pre_pca[weather_cols]

    pca = PCA(n_components=5)

    principal_components_weather = pca.fit_transform(df_weather)

    principal_df_weather = pd.DataFrame(
        data=principal_components_weather,
        columns=[
            'Weather_PC1',
            'Weather_PC2',
            'Weather_PC3',
            'Weather_PC4',
            'Weather_PC5'
        ]
    )

    df_pre_pca = df_pre_pca.drop(
        df_weather.columns.tolist(),
        axis=1
    )

    df_pre_pca = pd.concat(
        [
            df_pre_pca,
            principal_df_weather
        ],
        axis=1
    )

    df_pre_pca.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan,
        inplace=True
    )  # Replaces inf values with NaN values

    df_pre_pca.dropna(
        inplace=True,
        axis=0,
        how='any'
    )  # Drops all NaN values

    return df_pre_pca


"""
------------------ TASK 2 ------------------
"""


def linear_regression(df_processed: pd.DataFrame) -> None:
    """
    Linear regression model generation & testing
    :param df_processed: Data Frame containing data to build OLS Regression model
    """
    x, y, x_test, y_test = split_dataset(df_processed)  # Split data into training & testing

    model = sm.OLS(y, x)  # Generate Ordinary Least Squares Linear Regression model

    results = model.fit()  # Run OLS to get coefficients

    print(results.summary())

    prediction = results.predict(x_test)  # Generate predicted data from unseen testing data

    quality_graphs(prediction, y_test, 'Linear Regression')

    y_test = y_test.tolist()

    calculate_accuracy(prediction, y_test, 'Linear Regression')


"""
------------------ TASK 3 ------------------
"""


def artificial_neural_network(df_processed: pd.DataFrame) -> None:
    """
    Artificial neural network model generation, tuning & testing. This function uses Tensorflow to implement a neural
    network which predicts the daily household energy import.
    :param df_processed: Pre-processed data frame for training
    """
    train_features, train_labels, test_features, test_labels = split_dataset(df_processed)

    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=50,
        factor=3,
        directory='./checkpoints',
        project_name='acs341_assignment'
    )  # Build & configure the tuner object for tuning model hyperparameters

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )  # Tensorflow callback for stopping training early if improvement plateaus

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )  # Tensorflow callback for logging data to be viewable in Tensorboard

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
    )  # Tensorflow callback for reducing the optimiser learning rate on improvement plateau, for fine-tuning

    tuner.search(
        train_features,
        train_labels,
        epochs=100,
        validation_split=0.3,
        callbacks=[
            early_stopping_callback,
            reduce_lr_callback,
            tensorboard_callback
        ]
    )  # Run hyperparameter tuning on the neural network to generate & save the best settings to train on (or load old)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]  # Store best hyperparameters for building model

    ann_household_energy_model = tuner.hypermodel.build(best_hps)  # Build the ANN model with the generated settings

    history = ann_household_energy_model.fit(
        train_features,
        train_labels,
        validation_split=0.2,
        verbose=1,
        epochs=100,
        callbacks=[
            early_stopping_callback,
            reduce_lr_callback,
            tensorboard_callback
        ]
    )  # Train the model on the dataset, splitting into 20% validation data, 80% training data

    print(ann_household_energy_model.summary())  # Output the summary of the training, including all specified metrics

    plot_loss(history)
    ann_household_energy_model.evaluate(test_features, test_labels, verbose=1)  # Evaluate the model on the test data
    prediction = ann_household_energy_model.predict(test_features).flatten()  # Generate predictions from the test data

    quality_graphs(prediction, test_labels, 'Artificial Neural Network')  # Plot the test data performance

    test_labels = test_labels.tolist()
    calculate_accuracy(prediction, test_labels, 'Artificial Neural Network')


def build_model(hp) -> tf.keras.models.Model:
    """
    Builds Tensorflow Neural Network model using Keras API from Tensorflow

    :param hp: Model hyperparameters
    :return: Tensorflow Model
    """
    hp_units = hp.Int(
        'units',
        min_value=1,
        max_value=100,
        step=10
    )  # Define first layer units possibilities for training of hyperparameters

    model = keras.Sequential(
        [
            keras.layers.Dense(units=hp_units, activation='sigmoid'),
            keras.layers.Dense(64, activation='sigmoid'),
            keras.layers.Dense(1)
        ]
    )  # Definition of neural network model structure

    hp_learning_rate = hp.Choice(
        'learning_rate',
        values=[
            0.001,
            0.01,
            0.1,
            1.0
        ]
    )  # Possibilities for learning rate

    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=hp_learning_rate
        ),
        metrics=[
            'mae'
        ]
    )  # Compile the Tensorflow model & specify optimiser and desired metrics

    return model


def plot_loss(history) -> None:
    """
    Plots loss curve for both training data and validation data
    :param history: The stored metrics from the Keras model training
    """
    loss = (
            (history.history['loss'] + data_scaler[0][2]) *
            (data_scaler[0][1] - data_scaler[0][2])
    )[1:]  # Unscale
    val_loss = (
            (history.history['val_loss'] + data_scaler[0][2]) *
            (data_scaler[0][1] - data_scaler[0][2])
    )[1:]  # Unscale
    # Both loss and validation loss have had their first value dropped as loss starts very high, making
    # the rest of the graph unreadable
    print(loss)
    plt.plot(loss, label='Loss (MSE)')
    plt.plot(val_loss, label='Validation Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Error [kW^2]')
    plt.legend()
    plt.grid(True)
    plt.show()


"""
------------------ ADDITIONAL FOR FUNCTIONALITY ------------------
"""


def split_dataset(df_to_split: pd.DataFrame) -> tuple:
    """
    Splits data frame into training and test sets
    :param df_to_split: Full data set to be split
    :return: Training Input Set, Training Output Set, Testing Input Set, Testing Output Set
    """
    train_dataset = df_to_split.sample(frac=0.8, random_state=0, axis=0)  # Sample data for training (80%)
    test_dataset = df_to_split.drop(train_dataset.index)  # Drop training data to leave testing data

    # Split datasets into Inputs / Outputs
    x = train_dataset.iloc[:, 1:]
    x_test = test_dataset.iloc[:, 1:]

    y = train_dataset.iloc[:, 0]
    y_test = test_dataset.iloc[:, 0]

    # Add constant 1 column to inputs for offset compensation
    x = sm.add_constant(x)
    x_test = sm.add_constant(x_test)

    return x, y, x_test, y_test

def quality_graphs(prediction: list, test_labels: pd.DataFrame, title: str) -> None:
    """
    Plots quality graphs of prediction and test labels, allows for visualization of model performance
    :param prediction: Predicted data from generated model
    :param test_labels: Reserved unseen test output data
    :param title: Type of model used as string
    """
    prediction = (
            (prediction + data_scaler[0][2]) *
            (data_scaler[0][1] - data_scaler[0][2])
    )
    test_labels = (
            (test_labels + data_scaler[0][2]) *
            (data_scaler[0][1] - data_scaler[0][2])
    )

    regress = scipy.stats.linregress(test_labels, prediction)
    gradient = regress.slope
    offset = regress.intercept

    x = np.linspace(min(test_labels), max(test_labels), len(test_labels))
    y_optimal = x
    y_regress = (x * gradient) + offset

    plt.plot(test_labels, prediction, 'o',  zorder=1)
    plt.plot(x, y_optimal, '--', linewidth=4, zorder=2)
    plt.plot(x, y_regress, linewidth=4, zorder=3)
    plt.ylabel('Prediction Energy Requested From Grid [kW]')
    plt.xlabel('Actual Energy Requested From Grid [kW]')
    plt.title(title + ' Prediction vs Actual Energy Requested From Grid')
    plt.legend(['Values', 'Optimal', 'Fitted'])
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
    plt.hist(error, bins=50)
    plt.xlabel('Prediction Error [kW]')
    plt.ylabel('Count')
    plt.title(title + ' Prediction Error')
    plt.grid()
    plt.show()

def calculate_accuracy(prediction: list, test_labels: list, title: str) -> None:
    accuracy = 5

    prediction = (
            (prediction + data_scaler[0][2]) *
            (data_scaler[0][1] - data_scaler[0][2])
    )
    test_labels = (
            (test_labels + data_scaler[0][2]) *
            (data_scaler[0][1] - data_scaler[0][2])
    )

    mae = round(sklearn.metrics.mean_absolute_error(test_labels, prediction), accuracy)
    mse = round(sklearn.metrics.mean_squared_error(test_labels, prediction), accuracy)

    regress = scipy.stats.linregress(test_labels, prediction)
    gradient = round(regress.slope, accuracy)
    offset = round(regress.intercept, accuracy)

    pearson = round(np.corrcoef(test_labels, prediction)[0, 1], accuracy)

    print(f'\n\n{Fore.BLUE}{title.upper()} MAE: {mae}')
    print(f'{title.upper()} MSE: {mse}')
    print(f'{title.upper()} PEARSON CORRELATION COEFFICIENT: {pearson}')
    print(f'{title.upper()} FITTED COMPARISON EQUATION: PREDICTION = ({gradient} x TEST_LABELS) + {offset}{Fore.RESET}\n\n')



if __name__ == '__main__':
    """
    Main entry point for processing data, training and evaluating model
    """
    if not os.path.isfile(CSV_FILE_PATH):
        print(f'{Fore.RED}File {CSV_FILE_PATH} does not exist!\nPlease ensure CSV_FILE_PATH is set correctly'
              f' at the top of main_run.py{Fore.RESET}')
        exit(0)
    df = pd.read_csv(CSV_FILE_PATH)
    data_scaler = []
    df, data_scaler = data_processor(df, data_scaler)
    linear_regression(df)
    artificial_neural_network(df)
