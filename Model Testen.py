# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:25:19 2024

@author: nfritscher
"""

from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from datetime import datetime
import math as m
import os
import matplotlib.pyplot as plt
import zipfile

# data_list muss eine Liste von [Höhe, Breitengrad, Längengrad] des zu vorerzusagenden HSX des Standortes sein
from sklearn.metrics import mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred):
    # RMSE
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    # NRMSE
    nrmse = rmse / (y_true.max() - y_true.min())

    # R²
    r2 = r2_score(y_true, y_pred)

    return rmse, nrmse, r2


def aggregate_to_hourly(df, value_columns):
    # Gruppierung nach jeder sechsten Zeile
    a = 6 * 24
    hourly_df = df.groupby(df.index // a)[value_columns].mean()

    # Index zurücksetzen
    hourly_df.reset_index(drop=True, inplace=True)
    return hourly_df


def use_model(inpath, data_list, model_path, vienna=True, innsbruck=True, graz=True, klagenfurt=True, sonnblick=True, tl=True, rf=True, rr=False, rrm=False, ci=True, ff=False, dd=False, p=False,
              tag=False, monat=False, jahr=False, stunde=True, alt=True, long=True, elevation=False, toa=False, hsx=False):
    # Funktion zur Berechnung der Sonnendeklinationsfunktion nach DIN EN ISO 52010-1
    def fc_delta(n_day, leap_year=False):
        if not leap_year:
            Rdc = np.deg2rad(360 / 365 * n_day)
        else:
            Rdc = np.deg2rad(360 / 366 * n_day)

        return (0.33281 - 22.984 * np.cos(Rdc) - 0.3499 * np.cos(2 * Rdc) -
                0.1398 * np.cos(3 * Rdc) + 3.7872 * np.sin(Rdc) +
                0.03205 * np.sin(2 * Rdc) + 0.07187 * np.sin(3 * Rdc))

    print(alt)

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

    print(alt)

    # Sonnenhöhe/Elevation nach DIN EN ISO 52010-1
    def fc_alpha_sol(delta, omega, phi_w):
        return np.rad2deg(np.arcsin(np.sin(np.deg2rad(delta)) * np.sin(np.deg2rad(phi_w)) +
                                    np.cos(np.deg2rad(delta)) * np.cos(np.deg2rad(phi_w)) * np.cos(np.deg2rad(omega))))

    # Berechnung des Tages im Jahr aus Tag und Monat
    def get_day_of_year(row):
        date = datetime(row['Jahr'], row['Monat'], row['Tag'])
        return date.timetuple().tm_yday

    # base_folder = os.path.dirname(os.path.abspath(__file__))

    # Pfade zu den Trainingsdaten
    # training_data_Vienna = os.path.join(base_folder, "Data", "Mit HSX", "HS__W-5904-WIEN-HOHE_WARTE_10min.csv")
    # training_data_Graz = os.path.join(base_folder, "Data", "Mit HSX", "HS__St-16412-GRAZ-UNIVERSITAET_10min.csv")
    # training_data_Sonnblick = os.path.join(base_folder, "Data", "Mit HSX", "HS__S-15411-SONNBLICK_(TAWES)_10min.csv")
    # training_data_Klagenfurt = os.path.join(base_folder, "Data", "Mit HSX", "HS__K-20212+20211-KLAGENFURT-FLUGHAFEN_10min.csv")
    # training_data_Innsbruck = os.path.join(base_folder, "Data", "Mit HSX", "HS__T-11804-INNSBRUCK-FLUGPLATZ_10min.csv")

    training_data = {}

    data_list.insert(0, inpath)
    training_data["VorhersageStandort"] = data_list

    training_set = []
    print(alt)

    for data in training_data.values():
        location, altitude, latitude, longitude = data

        if location.endswith('.zip'):
            with zipfile.ZipFile(location, 'r') as zip_ref:
                zip_file_name = os.path.basename(location).replace('.zip', '.csv')
                with zip_ref.open(zip_file_name) as csv_file:
                    df = pd.read_csv(csv_file, sep=";", encoding='ISO-8859-1', on_bad_lines='skip')
        else:
            df = pd.read_csv(location, sep=";", encoding='ISO-8859-1', on_bad_lines='skip')

        df.replace('', np.nan, inplace=True)
        # print(df.head())
        print(df.columns)
        print(alt)

        df['Datetime'] = pd.to_datetime(df['Unnamed: 0'], errors='coerce')
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

        print(alt)

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

        # print(df.head(144))

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

        print(alt)
        df['TOA'] = df.apply(top_of_atmosphere_radiation, axis=1)

        def clearness_index(row):
            gsx = float(row['GSX'])
            toa = float(row['TOA'])
            if toa > 0:
                return gsx / toa
            else:
                return 0

        df['CI'] = df.apply(clearness_index, axis=1)

        # df = df.drop(['Unnamed: 0'], axis=1)
        # df = df.apply(pd.to_numeric, errors='coerce')
        # df = df.dropna()

    print(alt)
    training_set.append(df)

    df = pd.concat(training_set, ignore_index=True)

    print(df.columns)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # print(df.head(144))

    model = load_model(model_path)

    column_conditions = {
        "TL": tl,
        "RF": rf,
        "RR": rr,
        "RRM": rrm,
        "CI": ci,
        "TOA": toa,
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
        "HSX": hsx
    }
    print(alt)
    selected_columns = [column for column, is_active in column_conditions.items() if is_active]
    print(selected_columns)
    print(model.input_shape)  # Erwartete Form des Modells

    X = df[selected_columns]
    print("Selected columns for input:", selected_columns)
    print("Shape of X:", X.shape)

    # unused: 'Tag' und  'Jahr'
    # Vorhersagen
    predictions = model.predict(X)

    # Vorhersagen als neue Spalte hinzufügen
    df["HSX_Predicted"] = predictions

    df.loc[(df["CI"] == 0) & (df["Elevation"] == 0), "HSX_Predicted"] = 0

    # Gefilterte Daten mit Vorhersagen speichern
    output_path = os.path.join(os.path.dirname(inpath), "Predicted.csv")
    df.to_csv(output_path, index=False)
    print(f"Vorhersagen gespeichert unter: {output_path}")

    plt.figure(figsize=(10, 6))
    plt.scatter(df["HSX"], df["HSX_Predicted"], alpha=0.5, label="Vorhersage vs. Ist-Werte")
    plt.plot([df["HSX"].min(), df["HSX"].max()], [df["HSX"].min(), df["HSX"].max()], color="red", linestyle="--", label="Perfekte Vorhersage")
    plt.xlabel("Tatsächliche HSX-Werte")
    plt.ylabel("Vorhergesagte HSX-Werte")
    plt.title("Vergleich von vorhergesagten und tatsächlichen HSX-Werten")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Line-Plot für zeitliche Entwicklung
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["HSX"], label="Tatsächliche HSX-Werte", alpha=0.7)
    plt.plot(df.index, df["HSX_Predicted"], label="Vorhergesagte HSX-Werte", alpha=0.7)
    plt.xlabel("Index")
    plt.ylabel("HSX-Werte")
    plt.title("Tatsächliche und vorhergesagte HSX-Werte im Zeitverlauf")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Line-Plot: Zeitliche Entwicklung der ersten 240 Werte
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[:240], df["HSX"][:240], label="Tatsächliche HSX-Werte", alpha=0.7)
    plt.plot(df.index[:240], df["HSX_Predicted"][:240], label="Vorhergesagte HSX-Werte", alpha=0.7)
    plt.xlabel("Index")
    plt.ylabel("HSX-Werte")
    plt.title("Tatsächliche und vorhergesagte HSX-Werte im Zeitverlauf (erste 240 Werte)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Aggregation der Daten auf stündliche Basis
    hourly_data = aggregate_to_hourly(df, ["HSX", "HSX_Predicted"])

    # Visualisierung mit stündlichen Daten
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_data.index, hourly_data["HSX"], label="Tatsächliche HSX-Werte (stündlich)", alpha=0.7)
    plt.plot(hourly_data.index, hourly_data["HSX_Predicted"], label="Vorhergesagte HSX-Werte (stündlich)", alpha=0.7)
    plt.xlabel("Stunde")
    plt.ylabel("HSX-Werte")
    plt.title("Stündlich aggregierte tatsächliche und vorhergesagte HSX-Werte")
    plt.legend()
    plt.grid(True)
    plt.show()

    rmse, nrmse, r2 = calculate_metrics(df["HSX"], df["HSX_Predicted"])

    print(f"RMSE: {rmse}")
    print(f"NRMSE: {nrmse}")
    print(f"R²: {r2}")

    # Visualisierungen
    plt.figure(figsize=(10, 6))
    plt.scatter(df["HSX"], df["HSX_Predicted"], alpha=0.5, label="Vorhersage vs. Ist-Werte")
    plt.plot([df["HSX"].min(), df["HSX"].max()], [df["HSX"].min(), df["HSX"].max()], color="red", linestyle="--", label="Perfekte Vorhersage")
    plt.xlabel("Tatsächliche HSX-Werte")
    plt.ylabel("Vorhergesagte HSX-Werte")
    plt.title("Vergleich von vorhergesagten und tatsächlichen HSX-Werten")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Line-Plot für zeitliche Entwicklung
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["HSX"], label="Tatsächliche HSX-Werte", alpha=0.7)
    plt.plot(df.index, df["HSX_Predicted"], label="Vorhergesagte HSX-Werte", alpha=0.7)
    plt.xlabel("Index")
    plt.ylabel("HSX-Werte")
    plt.title("Tatsächliche und vorhergesagte HSX-Werte im Zeitverlauf")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Stündliche Aggregation und Visualisierung
    hourly_data = aggregate_to_hourly(df, ["HSX", "HSX_Predicted"])

    plt.figure(figsize=(12, 6))
    plt.plot(hourly_data.index, hourly_data["HSX"], label="Tatsächliche HSX-Werte (stündlich)", alpha=0.7)
    plt.plot(hourly_data.index, hourly_data["HSX_Predicted"], label="Vorhergesagte HSX-Werte (stündlich)", alpha=0.7)
    plt.xlabel("Stunde")
    plt.ylabel("HSX-Werte")
    plt.title("Stündlich aggregierte tatsächliche und vorhergesagte HSX-Werte")
    plt.legend()
    plt.grid(True)
    plt.show()


def main_script():
    # Testen wie gut für den Standort Innsbruck HSX vorhergesagt wird
    base_folder = os.path.dirname(os.path.abspath(__file__))
    vienna = True
    innsbruck = True
    graz = True
    klagenfurt = True
    sonnblick = True
    tl = True
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
    stunde = True
    alt = True
    long = True
    elevation = False
    toa = True
    hsx = False

    inpath = os.path.join(base_folder, "Data", "Ohne HSX", "HS__T-11804-INNSBRUCK-FLUGPLATZ", "2020.zip")
    data_list = [581, 47.2603, 11.3439]  # Höhe, Breitengrad, Längengrad
    #    model_path = os.path.join(base_folder, "hsx_0.001_sigmoid_sigmoid_32_100.keras")
    model_path = os.path.join(base_folder, "hsx_0.0001_sigmoid_sigmoid_32_10.keras")

    use_model(inpath, data_list, model_path, vienna, innsbruck, graz, klagenfurt, sonnblick, tl, rf, rr, rrm, ci, ff, dd, p, tag, monat, jahr, stunde, alt, long, elevation, toa, hsx)


if __name__ == '__main__':
    main_script()
