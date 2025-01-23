import feature_engineering as fen
import pandas as pd
import time


def preprocessing(locations=['Vienna'], save_to=None):
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

        # df = pd.read_csv(location, sep=";", encoding='ISO-8859-1', on_bad_lines='skip')
        # df_split = df.iloc[:, 0].str.split(',', expand=True)
        # df_split.columns = ['Datum', 'TL', 'TL_FLAG', 'RF', 'RF_FLAG', 'RR', 'RR_FLAG', 'RRM', 'RRM_FLAG',
        #                     'GSX', 'GSX_FLAG', 'FF', 'FF_FLAG', 'DD', 'DD_FLAG', 'P', 'P_FLAG', 'HSX', 'HSX_FLAG']
        # df = pd.concat([df_split, df.iloc[:, 1:]], axis=1)
        # df.replace('', np.nan, inplace=True)
        # df['Datetime'] = pd.to_datetime(df['Datum'], errors='coerce')
        # df = df.dropna(subset=['Datetime'])
        df = pd.read_csv(path_and_file, index_col=0)
        df.index = pd.to_datetime(df.index)

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
        df['CI'] = df.apply(fen.clearness_index, axis=1)
        training_set.append(df)

    df = pd.concat(training_set, ignore_index=True)
    return df


def train_model(X, y):
    pass


def main_script(processed_data='processed_vienna.csv', preproc=False, write_preproc=False, training=True):
    # locations = ['Vienna', 'Klagenfurt', 'Sonnblick', 'Innsbruck', 'Graz']

    if preproc:
        locations = ['Vienna']
        print(f'locations to process: {locations}')

        time_start = time.time()
        print('preprocessing... ', end='', flush=True)
        df = preprocessing(locations=locations)
        print(f'{time.time() - time_start:.2f} sec')

        if write_preproc:
            time_start = time.time()
            print(f'writing to file {processed_data}... ', end='', flush=True)
            df.to_csv(processed_data)
            print(f'{time.time() - time_start:.2f} sec')
    else:
        time_start = time.time()
        print(f'openinng {processed_data}... ', end='', flush=True)
        df = pd.read_csv(processed_data, index_col=0)
        print(f'{time.time() - time_start:.2f} sec')

    if training:
        time_start = time.time()
        print(f'training model... ', end='', flush=True)
        list_of_features = ["TL", "RF", "RR", "RRM", "CI", "GSX", "FF", "DD", "P", "Tag", "Monat", "Jahr", "Stunde",
                            "Altitude", "Longitude", "Elevation", "TOA"]

        train_model(X=df[list_of_features], y=df['HSX'])
        print(f'{time.time() - time_start:.2f} sec')


if __name__ == '__main__':
    main_script()
