import numpy as np
from datetime import datetime


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
    date = datetime(int(row['Jahr']), int(row['Monat']), int(row['Tag']))
    return date.timetuple().tm_yday


def fc_alpha_sol_wrap(row):
    leap_years = [2004, 2008, 2012, 2016, 2020]

    day = row['day_of_year']
    hour = row['Stunde'] + row['Minute'] / 60 + 0.5
    TZ = 1
    lambda_w = row['Longitude']
    phi_w = row['Latitude']
    alpha_min = 0
    year = row.name.year

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


def top_of_atmosphere_radiation(row):
    n_day = row['day_of_year']
    elevation = row['Elevation']
    I_sc = 1361

    if elevation > 0:
        theta = np.radians(90 - elevation)  # Zenith-Winkel
        toa = I_sc * ((1 + 0.034 * np.cos((2 * np.pi * n_day) / 365)) * np.cos(theta))
    else:
        toa = 0

    return toa


def clearness_index(row):
    gsx = float(row['GSX'])
    toa = float(row['TOA'])
    if toa > 0:
        return gsx / toa
    else:
        return 0


def calculate_sun_azimuth(row):
    leap_years = [2004, 2008, 2012, 2016, 2020]
    day = row['day_of_year']
    hour = row['Stunde'] + row['Minute'] / 60  + 0.5
    TZ = 1
    lambda_w = row['Longitude']
    phi_w = row['Latitude']
    alpha_sol = row["Elevation"]
    year = row.name.year
    leap_year = year in leap_years
    
    delta = fc_delta(day, leap_year)
    t_eq = fc_t_eq(day)
    t_shift = fc_t_shift(TZ, lambda_w)
    t_sol = fc_t_sol(t_eq, t_shift, hour)
    omega = fc_omega(t_sol)
    
    # Berechnung der Hilfsvariablen gemäß den Gleichungen (13) bis (15) in DIN EN ISO 52010-1
    sin_phi_sol_aux1 = np.cos(np.radians(delta)) * np.sin(np.radians(180 - omega))/(np.cos(np.arcsin(np.sin(np.radians(alpha_sol)))))
    
    cos_phi_sol_aux1 = (np.cos(np.radians(phi_w)) * np.sin(np.radians(delta)) + np.sin(np.radians(phi_w)) * np.cos(np.radians(delta)) * np.cos(np.radians(180 - omega)))/(np.cos(np.arcsin(np.sin(np.radians(alpha_sol)))))
    
    phi_sol_aux2 = np.degrees(np.arcsin(np.cos(np.radians(delta)) * np.sin(np.radians(180 - omega))))/((np.cos(np.arcsin(np.sin(np.radians(alpha_sol))))))

    # Berechnung des Sonnenazimutwinkels gemäß Gleichung (16) in DIN EN ISO 52010-1
    if sin_phi_sol_aux1 >= 0 and cos_phi_sol_aux1 > 0:
        phi_sol = 180 - phi_sol_aux2
    elif cos_phi_sol_aux1 < 0:
        phi_sol = phi_sol_aux2
    else:
        phi_sol = - (180 + phi_sol_aux2)
        
    # if phi_sol <= 0:
    #     phi_sol = 180 - phi_sol
    # else:
    #     phi_sol = (180 - phi_sol)

    phi_sol = 180 - phi_sol   
    return phi_sol