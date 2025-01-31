import time
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Nadam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from tensorflow.keras.metrics import RootMeanSquaredError
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score

import feature_engineering as fen


def preprocess(locations=['Vienna', 'Sonnblick', 'Klagenfurt', 'Graz', 'Innsbruck'], save_to=None, date_limits=None):
    base_path = r'./Data/Mit HSX'
    training_data_paths = {
        'Vienna': f'{base_path}/HS__W-5904-WIEN-HOHE_WARTE_10min.csv',
        'Graz': f'{base_path}/HS__St-16412-GRAZ-UNIVERSITAET_10min.csv',
        'Sonnblick': f'{base_path}/HS__S-15411-SONNBLICK_(TAWES)_10min.csv',
        'Klagenfurt': f'{base_path}/HS__K-20212+20211-KLAGENFURT-FLUGHAFEN_10min.csv',
        'Innsbruck': f'{base_path}/HS__T-11804-INNSBRUCK-FLUGPLATZ_10min.csv'
    }

    location_data = {
        'Vienna': [training_data_paths['Vienna'], 197., 48.2494, 16.3561],
        'Innsbruck': [training_data_paths['Innsbruck'], 581., 47.2603, 11.3439],
        'Graz': [training_data_paths['Graz'], 353., 47.0805, 15.4487],
        'Klagenfurt': [training_data_paths['Klagenfurt'], 449., 46.6422, 14.3377],
        'Sonnblick': [training_data_paths['Sonnblick'], 3106., 47.0544, 12.9573]
    }

    training_data = dict()
    for loc in locations:
        training_data[loc] = location_data[loc]

    training_set = list()
    for loc in locations:
        path_and_file, altitude, latitude, longitude = training_data[loc]
        df = pd.read_csv(path_and_file, index_col=0)
        df.index = pd.to_datetime(df.index)
        if date_limits is not None:
            df = df[(df.index >= date_limits[0]) & (df.index < date_limits[1])]

        df['Tag'] = df.index.day
        df['Monat'] = df.index.month
        df['Jahr'] = df.index.year
        df['Stunde'] = df.index.hour
        df['Minute'] = df.index.minute
        df['Sekunde'] = df.index.second
        # df['day_of_year_old'] = df.apply(fen.get_day_of_year, axis=1)
        df['day_of_year'] = df.index.dayofyear

        df['Altitude'] = altitude
        df['Latitude'] = latitude
        df['Longitude'] = longitude

        #  this takes forever - so lets skip it for now
        df['Elevation'] = df.apply(fen.fc_alpha_sol_wrap, axis=1)
        df['TOA'] = df.apply(fen.top_of_atmosphere_radiation, axis=1)
        df['time_distance'] = df.apply(fen.calculate_time_distance, axis=1)
        df = fen.calculate_gsx_3h_mean(df)
        df['CI'] = df.apply(fen.clearness_index, axis=1)
        df["Azimuth"] = df.apply(fen.calculate_sun_azimuth,axis=1)
        training_set.append(df)

    df = pd.concat(training_set)
    return df


def preprocessing_script(locations=tuple(['Vienna', 'Sonnblick', 'Klagenfurt', 'Graz', 'Innsbruck']), save_as=None, date_limits=None):
    # print(f'locations to process: {locations}')
    time_start = time.time()
    print('preprocessing... ', end='', flush=True)
    df = preprocess(locations=locations, date_limits=date_limits)
    print(f'{time.time() - time_start:.2f} sec')

    if save_as is not None:
        time_start = time.time()
        print(f'writing to file {save_as}... ', end='', flush=True)
        df.to_csv(save_as)
        print(f'{time.time() - time_start:.2f} sec')


def training_script(df, target='HSX',
                    list_of_features=None,
                    learning_rate=0.001, batch_size=64, epochs=1000, test_size=0.3,
                    show_lossplot=True, save_as='trained_model.keras',
                    list_of_layers=None):
    time_start = time.time()
    print(f'training model... ', end='', flush=True)
    if list_of_features is None:
        list_of_features = df.columns.pop(target)
    # scaler_x = MinMaxScaler()
    # x_scaled = scaler_x.fit_transform(X)
    # X_train, x_val, y_train, y_val = train_test_split(x_scaled, y, test_size=test_size, random_state=42)

    X = df[list_of_features]
    y = df[target]
    X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    for layer in list_of_layers:
        model.add(layer)
    model.add(Dense(1))
    model.compile(optimizer=Nadam(learning_rate), loss="mse", metrics=[RootMeanSquaredError()])

    model_checkpoint = ModelCheckpoint(save_as, save_best_only=True, monitor='val_loss')
    early_stopping = EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True)

    history = model.fit(X_train, y_train, batch_size, epochs, validation_data=(x_val, y_val),
                        callbacks=[model_checkpoint, early_stopping])
    model.save(save_as)
    print(f"model saved as {save_as}")
    print(f'{time.time() - time_start:.2f} sec')

    if show_lossplot:
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss During Training')
        plt.show()


def main_script():
    training_data_file = 'processed_vienna_2016.csv'
    training_period = ('2003', '2023')
    evaluation_data_file = 'processed_vienna_2024.csv'
    evaluation_period = ('2023', '2024')

    model_file = 'trained_model.keras'
    target = 'HSX'
    list_of_locations = ['Vienna', 'Sonnblick', 'Klagenfurt', 'Graz', 'Innsbruck']
    preprocessing_script(locations=list_of_locations, save_as=training_data_file, date_limits=training_period)
    #preprocessing_script(save_as=evaluation_data_file, date_limits=evaluation_period)

    time_start = time.time()
    print(f'opening {training_data_file}... ', end='', flush=True)
    df_train = pd.read_csv(training_data_file, index_col=0)
    # df_train.index = pd.to_datetime(df_train.index)
    print(f'{time.time() - time_start:.2f} sec')
    print(df)
    list_of_features = ["TL", "RF", "RR", "RRM", "CI", "GSX", "FF", "DD", "P", "Tag", "Monat", "Jahr", "time_distance",
                        "Altitude", "Longitude", "Elevation", "Azimuth", "TOA", 'GSX_3h_mean']
    list_of_layers = [Dense(64, 'relu'),
                      Dense(32, 'relu'),
                      Dense(16, 'relu')]
    training_script(df=df_train, list_of_features=list_of_features, list_of_layers=list_of_layers, target=target,
                    save_as=model_file, epochs=3)

    time_start = time.time()
    print(f'opening {evaluation_data_file}... ', end='', flush=True)
    df_eval = pd.read_csv(evaluation_data_file, index_col=0)
    df_eval.index = pd.to_datetime(df_eval.index)
    print(f'{time.time() - time_start:.2f} sec')
    X_new = df_eval[list_of_features][(df_eval.index >= evaluation_period[0]) & (df_eval.index < evaluation_period[1])]
    model = load_model(model_file)
    predictions = model.predict(X_new)
    df_eval['predictions'] = predictions
    print(f'coef of determination: {r2_score(df_eval["predictions"], df_eval[target]):.3f}')
    plt.plot(df_eval['predictions'], label='predictions')
    plt.plot(df_eval[target], label='measurements')
    plt.grid()
    plt.legend()
    plt.show(block=True)


if __name__ == '__main__':
    main_script()
