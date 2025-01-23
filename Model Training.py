from tensorflow.keras.optimizers import Nadam
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math as m
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf


def train_model(outpath="variabler Name", learning_rate=0.0001, batch_size=64, epochs=10, activation="sigmoid", test_size=0.3, vienna=True, innsbruck=True, graz=True, klagenfurt=True, sonnblick=True,
                tl=True, rf=True, rr=True, rrm=True, ci=True, gsx=True, ff=True, toa=True, dd=True, p=True, tag=True, monat=True, jahr=True, stunde=True, alt=True, long=True, elevation=True,
                hsx=False, target="HSX", show_lossplot=True, save_preproc_as=None):
    if outpath == "variabler Name":
        outpath = os.path.abspath(f"./hsx_{learning_rate}_{activation}_{batch_size}_{epochs}.keras")
    else:
        outpath = os.path.abspath(outpath)

    model_name = f"hsx_{learning_rate}_{activation}_{'sigmoid'}_{batch_size}_{epochs}.keras"

    # Funktion zur Berechnung der Sonnendeklinationsfunktion nach DIN EN ISO 52010-1
    def fc_delta(n_day, leap_year=False):
        if not leap_year:
            Rdc = np.deg2rad(360 / 365 * n_day)
        else:
            Rdc = np.deg2rad(360 / 366 * n_day)

        return (0.33281 - 22.984 * np.cos(Rdc) - 0.3499 * np.cos(2 * Rdc) -
                0.1398 * np.cos(3 * Rdc) + 3.7872 * np.sin(Rdc) +
                0.03205 * np.sin(2 * Rdc) + 0.07187 * np.sin(3 * Rdc))

    # Zeitgleichung nach DIN EN ISO 52010-1
    def fc_t_eq(n_day):
        if n_day < 21:
            return 2.6 + 0.44 * n_day
        elif 21 <= n_day < 136:
            return 5.2 + 9.0 * np.cos(np.deg2rad((n_day - 43) * 0.0357 * 180 / np.pi))
        elif 136 <= n_day < 241:
            return 1.4 - 5.0 * np.cos(np.deg2rad((n_day - 135) * 0.0449 * 180 / np.pi))
        elif 241 <= n_day < 336:
            return -6.3 - 10 * np.cos(np.deg2rad((n_day - 306) * 0.036 * 180 / np.pi))
        elif n_day >= 336:
            return 0.45 * (n_day - 359)
        else:
            raise ValueError("Invalid n_day")

    # Sonnenzeit nach DIN EN ISO 52010-1
    def fc_t_shift(TZ, lambda_w):
        return TZ - lambda_w / 15

    def fc_t_sol(t_eq, t_shift, n_hour):
        return n_hour - t_eq / 60 - t_shift

    def fc_omega(t_sol):
        omega = 180 / 12 * (12.5 - t_sol)
        if omega > 180:
            return omega - 360
        elif omega < -180:
            return omega + 360
        else:
            return omega

    # Sonnenhöhe/Elevation nach DIN EN ISO 52010-1
    def fc_alpha_sol(delta, omega, phi_w):
        return np.rad2deg(np.arcsin(np.sin(np.deg2rad(delta)) * np.sin(np.deg2rad(phi_w)) +
                                    np.cos(np.deg2rad(delta)) * np.cos(np.deg2rad(phi_w)) * np.cos(np.deg2rad(omega))))

    # Berechnung des Tages im Jahr aus Tag und Monat
    def get_day_of_year(row):
        date = datetime(row['Jahr'], row['Monat'], row['Tag'])
        return date.timetuple().tm_yday

    base_folder = os.path.dirname(os.path.abspath(__file__))

    # Pfade zu den Trainingsdaten
    training_data_Vienna = os.path.join(base_folder, "Data", "Mit HSX", "HS__W-5904-WIEN-HOHE_WARTE_10min.csv")
    training_data_Graz = os.path.join(base_folder, "Data", "Mit HSX", "HS__St-16412-GRAZ-UNIVERSITAET_10min.csv")
    training_data_Sonnblick = os.path.join(base_folder, "Data", "Mit HSX", "HS__S-15411-SONNBLICK_(TAWES)_10min.csv")
    training_data_Klagenfurt = os.path.join(base_folder, "Data", "Mit HSX", "HS__K-20212+20211-KLAGENFURT-FLUGHAFEN_10min.csv")
    training_data_Innsbruck = os.path.join(base_folder, "Data", "Mit HSX", "HS__T-11804-INNSBRUCK-FLUGPLATZ_10min.csv")

    location_data = {
        "Vienna": [training_data_Vienna, 197, 48.2494, 16.3561],
        "Innsbruck": [training_data_Innsbruck, 581, 47.2603, 11.3439],
        "Graz": [training_data_Graz, 353, 47.0805, 15.4487],
        "Klagenfurt": [training_data_Klagenfurt, 449, 46.6422, 14.3377],
        "Sonnblick": [training_data_Sonnblick, 3.106, 47.0544, 12.9573]
    }

    training_data = {}

    if vienna:
        training_data["Vienna"] = location_data["Vienna"]

    if innsbruck:
        training_data["Innsbruck"] = location_data["Innsbruck"]

    if graz:
        training_data["Graz"] = location_data["Graz"]

    if klagenfurt:
        training_data["Klagenfurt"] = location_data["Klagenfurt"]

    if sonnblick:
        training_data["Sonnblick"] = location_data["Sonnblick"]

    training_set = []

    for data in training_data.values():
        location, altitude, latitude, longitude = data
        df = pd.read_csv(location, sep=";", encoding='ISO-8859-1', on_bad_lines='skip')
        df_split = df.iloc[:, 0].str.split(',', expand=True)
        df_split.columns = ['Datum', 'TL', 'TL_FLAG', 'RF', 'RF_FLAG', 'RR', 'RR_FLAG', 'RRM', 'RRM_FLAG',
                            'GSX', 'GSX_FLAG', 'FF', 'FF_FLAG', 'DD', 'DD_FLAG', 'P', 'P_FLAG', 'HSX', 'HSX_FLAG']
        df = pd.concat([df_split, df.iloc[:, 1:]], axis=1)
        df.replace('', np.nan, inplace=True)
        df['Datetime'] = pd.to_datetime(df['Datum'], errors='coerce')
        df = df.dropna(subset=['Datetime'])

        df['Tag'] = df['Datetime'].dt.day
        df['Monat'] = df['Datetime'].dt.month
        df['Jahr'] = df['Datetime'].dt.year
        df['Stunde'] = df['Datetime'].dt.hour
        df['Minute'] = df['Datetime'].dt.minute
        df['Sekunde'] = df['Datetime'].dt.second

        df['Altitude'] = data[1]
        df['Latitude'] = data[2]
        df['Longitude'] = data[3]

        df['day_of_year'] = df.apply(get_day_of_year, axis=1)

        # Berechnung der Sonnenelevation
        def fc_alpha_sol_wrap(row):
            leap_years = [2004, 2008, 2012, 2016, 2020]

            day = row['day_of_year']
            hour = row['Stunde'] + row['Minute'] / 60 + 0.5
            TZ = 1
            lambda_w = row['Longitude']
            phi_w = row['Latitude']
            alpha_min = 0
            year = row['Datetime'].year

            leap_year = year in leap_years

            delta = fc_delta(day, leap_year)
            t_eq = fc_t_eq(day)
            t_shift = fc_t_shift(TZ, lambda_w)
            t_sol = fc_t_sol(t_eq, t_shift, hour)
            omega = fc_omega(t_sol)
            alpha_sol = fc_alpha_sol(delta, omega, phi_w)

            if alpha_sol < alpha_min:
                alpha_sol = 0

            return alpha_sol

        df['Elevation'] = df.apply(fc_alpha_sol_wrap, axis=1)

        def top_of_atmosphere_radiation(row):
            n_day = row['day_of_year']
            elevation = row['Elevation']
            I_sc = 1361

            if elevation > 0:
                theta = m.radians(90 - elevation)  # Zenith-Winkel
                toa = I_sc * ((1 + 0.034 * m.cos((2 * m.pi * n_day) / 365)) * m.cos(theta))
            else:
                toa = 0

            return toa

        df['TOA'] = df.apply(top_of_atmosphere_radiation, axis=1)

        def clearness_index(row):
            gsx = float(row['GSX'])
            toa = float(row['TOA'])
            if toa > 0:
                return gsx / toa
            else:
                return 0

        df['CI'] = df.apply(clearness_index, axis=1)

        df.loc[(df['CI'] == 0) & (df['Elevation'] == 0), 'HSX'] = 0
        df = df.drop(['Datum'], axis=1)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        flag_columns = [col for col in df.columns if '_FLAG' in col]
        df = df[(df[flag_columns] <= 32).all(axis=1)]

        print("Zeilenanzahl der Messwerte:", df.shape[0])
        training_set.append(df)

    df = pd.concat(training_set, ignore_index=True)
    print("Zeilenanzahl der Messwerte:", df.shape[0])

    print(df.columns)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(df.head(144))

    if save_preproc_as is not None:
        df.to_csv(save_preproc_as)

    def evaluate_model(model, x_val, y_val):
        y_pred = model.predict(x_val, verbose=0).flatten()
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        nrmse = rmse / (y_val.max() - y_val.min()) if y_val.max() != y_val.min() else 0
        r2 = r2_score(y_val, y_pred) if len(y_val) > 1 else None
        correlation_matrix = np.corrcoef(y_val, y_pred.ravel()) if len(y_val) > 1 else None
        R = correlation_matrix[0, 1] if correlation_matrix is not None else None
        return {"RMSE": rmse, "MAE": mae, "MSE": mse, "nRMSE": nrmse, "R²": r2, "R": R}

    def save_model_and_metrics(model, metrics, directory):

        model_path = os.path.join(base_folder, model_name)
        model.save(model_path)
        print(f"Modell gespeichert unter: {model_path}")

        metrics_path = os.path.join(base_folder, "metrics.txt")
        with open(metrics_path, "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        print(f"Metriken gespeichert unter: {metrics_path}")

    column_conditions = {
        "TL": tl,
        "RF": rf,
        "RR": rr,
        "RRM": rrm,
        "CI": ci,
        "GSX": gsx,
        "FF": ff,
        "DD": dd,
        "P": p,
        "Tag": tag,
        "Monat": monat,
        "Jahr": jahr,
        "Stunde": stunde,
        "Altitude": alt,
        "Longitude": long,
        "Elevation": elevation,
        "TOA": toa,
        "HSX": hsx
    }

    selected_columns = [column for column, is_active in column_conditions.items() if is_active]

    x = df[selected_columns]
    y = df[target]

    scaler_x = MinMaxScaler()
    x_scaled = scaler_x.fit_transform(x)

    X_train, x_val, y_train, y_val = train_test_split(x_scaled, y, test_size=test_size, random_state=42)

    model = Sequential()

    model.add(Input(shape=(X_train.shape[1],)))

    model.add(Dense(64, activation))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation))
    model.add(Dropout(0.3))

    model.add(Dense(16, activation))
    model.add(Dropout(0.3))

    model.add(Dense(1))

    model.compile(optimizer=Nadam(learning_rate), loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])

    model_checkpoint = ModelCheckpoint(model_name, save_best_only=True, monitor='val_loss')
    early_stopping = EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True)

    history = model.fit(X_train, y_train, batch_size, epochs, validation_data=(x_val, y_val),
                        callbacks=[model_checkpoint, early_stopping])

    if show_lossplot == True:
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss During Training')
        plt.show()

    metrics = evaluate_model(model, x_val, y_val)

    save_model_and_metrics(model, metrics, outpath)
    return ("Model wurde in", outpath, "gespeichert")


def main_script():
    # Training mit allen Datensätzen für Diffusstrahlung

    outpath = "variabler Name"
    learning_rate = 0.0001
    batch_size = 32
    epochs = 10
    activation = "sigmoid"
    test_size = 0.3
    vienna = True
    innsbruck = False
    graz = False
    klagenfurt = False
    sonnblick = False
    target = "HSX"
    show_lossplot = False
    tl = False
    gsx = True
    rf = True
    rr = False
    rrm = False
    ci = True
    ff = False
    dd = False
    p = False
    tag = False
    monat = False
    jahr = False
    stunde = False
    alt = False
    long = False
    elevation = False
    toa = True,
    toa = True
    hsx = False

    train_model(outpath, learning_rate, batch_size, epochs, activation, test_size, vienna, innsbruck, graz, klagenfurt, sonnblick, tl, rf, rr, gsx, rrm, ci, ff, dd, p, tag, monat, jahr, stunde, alt,
                long, elevation, toa, hsx, target, show_lossplot, save_preproc_as='noah_preproc.csv')


if __name__ == '__main__':
    main_script()
