import pickle
import inflection
import pandas as pd
import numpy as np
import datetime
from pathlib import Path

_here = Path(__file__).resolve().parent
models_path_parameters = _here.parent.parent / 'models' / 'parameters'

class Rossmann(object):
    def __init__(self):
        self.competition_distance_scaler = pickle.load(open(models_path_parameters / 'rs_distance.pkl',  'rb'))
        self.competition_time_month      = pickle.load(open(models_path_parameters / 'rs_time.pkl',      'rb'))
        self.promo_time_week             = pickle.load(open(models_path_parameters / 'mms_promo.pkl',    'rb'))
        self.year                        = pickle.load(open(models_path_parameters / 'mms_year.pkl',     'rb'))
        self.store_type                  = pickle.load(open(models_path_parameters / 'store_type.pkl',   'rb'))

    def data_cleaning(self, df1):
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
                    'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

        # Select only expected columns before renaming to guard against extra payload columns
        df1 = df1[cols_old].copy()

        snakecase = lambda x: inflection.underscore(x)
        cols_new = list(map(snakecase, cols_old))
        df1.columns = cols_new

        df1['date'] = pd.to_datetime(df1['date'])

        # pd.isna handles both float NaN and Python None (JSON null), unlike math.isnan
        df1['competition_distance'] = df1['competition_distance'].apply(
            lambda x: 200000.0 if pd.isna(x) else x)

        df1['competition_open_since_month'] = df1.apply(
            lambda x: x['date'].month if pd.isna(x['competition_open_since_month'])
                      else x['competition_open_since_month'], axis=1)

        df1['competition_open_since_year'] = df1.apply(
            lambda x: x['date'].year if pd.isna(x['competition_open_since_year'])
                      else x['competition_open_since_year'], axis=1)

        # int() cast avoids UInt32 dtype from isocalendar() mixing with float in the column
        df1['promo2_since_week'] = df1.apply(
            lambda x: int(x['date'].isocalendar().week) if pd.isna(x['promo2_since_week'])
                      else x['promo2_since_week'], axis=1)

        df1['promo2_since_year'] = df1.apply(
            lambda x: x['date'].year if pd.isna(x['promo2_since_year'])
                      else x['promo2_since_year'], axis=1)

        # 'Feb' matches Rossmann's PromoInterval strings; original had 'Fev' (Portuguese)
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                     7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

        df1['promo_interval'] = df1['promo_interval'].fillna('0')
        df1['month_map'] = df1['date'].dt.month.map(month_map)
        df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(
            lambda x: 0 if x['promo_interval'] == '0'
                      else 1 if x['month_map'] in x['promo_interval'].split(',')
                      else 0,
            axis=1)

        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year']  = df1['competition_open_since_year'].astype(int)
        df1['promo2_since_week']            = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year']            = df1['promo2_since_year'].astype(int)

        return df1

    def feature_engineering(self, df2):
        df2['year']         = df2['date'].dt.year
        df2['month']        = df2['date'].dt.month
        df2['day']          = df2['date'].dt.day
        # astype(int) converts UInt32 returned by isocalendar() to plain int64
        df2['week_of_year'] = df2['date'].dt.isocalendar().week.astype(int)
        df2['year_week']    = df2['date'].dt.strftime('%Y-%W')

        df2['competition_since'] = df2.apply(
            lambda x: datetime.datetime(year=x['competition_open_since_year'],
                                        month=x['competition_open_since_month'], day=1), axis=1)
        # .dt.days extracts integer days from the Timedelta Series before dividing,
        # avoiding the float result that broke .days attribute access in the original
        df2['competition_time_month'] = (df2['date'] - df2['competition_since']).dt.days // 30

        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply(
            lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))
        df2['promo_time_week'] = (df2['date'] - df2['promo_since']).dt.days // 7

        df2['assortment'] = df2['assortment'].apply(
            lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

        df2['state_holiday'] = df2['state_holiday'].apply(
            lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b'
                      else 'christmas' if x == 'c' else 'regular_day')

        df2 = df2[df2['open'] != 0]

        cols_drop = ['open', 'promo_interval', 'month_map']
        df2 = df2.drop(cols_drop, axis=1)

        return df2

    def data_preparation(self, df5):
        # API context: no train/test split needed; apply transforms directly to the full input
        X_test = df5.copy()

        X_test['competition_distance']   = self.competition_distance_scaler.transform(X_test[['competition_distance']].values)
        X_test['competition_time_month'] = self.competition_time_month.transform(X_test[['competition_time_month']].values)
        X_test['promo_time_week']        = self.promo_time_week.transform(X_test[['promo_time_week']].values)
        X_test['year']                   = self.year.transform(X_test[['year']].values)

        X_test = pd.get_dummies(X_test, prefix=['state_holiday'], columns=['state_holiday'], dtype=int)
        # Guarantee all OHE columns exist regardless of which categories appear in this batch
        for col in ['state_holiday_christmas', 'state_holiday_easter_holiday',
                    'state_holiday_public_holiday', 'state_holiday_regular_day']:
            if col not in X_test.columns:
                X_test[col] = 0

        X_test['store_type'] = self.store_type.transform(X_test['store_type'])

        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        X_test['assortment'] = X_test['assortment'].map(assortment_dict)

        # Vectorized cyclic encoding (no per-row apply overhead)
        X_test['day_of_week_sin']  = np.sin(X_test['day_of_week']  * (2. * np.pi / 7))
        X_test['day_of_week_cos']  = np.cos(X_test['day_of_week']  * (2. * np.pi / 7))
        X_test['month_sin']        = np.sin(X_test['month']         * (2. * np.pi / 12))
        X_test['month_cos']        = np.cos(X_test['month']         * (2. * np.pi / 12))
        X_test['day_sin']          = np.sin(X_test['day']           * (2. * np.pi / 30))
        X_test['day_cos']          = np.cos(X_test['day']           * (2. * np.pi / 30))
        X_test['week_of_year_sin'] = np.sin(X_test['week_of_year']  * (2. * np.pi / 52))
        X_test['week_of_year_cos'] = np.cos(X_test['week_of_year']  * (2. * np.pi / 52))

        cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance',
                         'competition_open_since_month', 'competition_open_since_year',
                         'promo2', 'promo2_since_week', 'promo2_since_year',
                         'competition_time_month', 'promo_time_week',
                         'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
                         'day_sin', 'day_cos', 'week_of_year_sin', 'week_of_year_cos']

        return X_test[cols_selected]

    def get_prediction(self, model, original_data, test_data):
        pred = model.predict(test_data)
        # Index alignment: feature_engineering may drop open=0 rows, so len(pred) <= len(original_data)
        original_data = original_data.copy()
        original_data.loc[test_data.index, 'prediction'] = np.expm1(pred)
        return original_data.to_json(orient='records', date_format='iso')
