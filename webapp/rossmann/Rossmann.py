#==================================
# Import Library
#==================================

import pickle
import inflection
import pandas as pd
import numpy as np
import datetime
from pathlib import Path

#==================================
# Paths
#==================================

_here = Path(__file__).resolve().parent
models_path_parameters = _here.parent / 'parameter'

#===================================
# Pipeline de Predição
#===================================

class Rossmann(object):

    #Função 1: Carregamento dos Encoders e Scalers Treinados
    def __init__(self):
        """
        Carrega os encoders e scalers ajustados no notebook de treino, para
        preparar os dados de entrada da API exatamente como o modelo espera.

        Parâmetros
        ----------
        Nenhum — os 5 arquivos .pkl são lidos de parameter/, relativamente a
        este módulo.

        Retorna
        -------
        None — os objetos carregados ficam disponíveis nos atributos da
        instância: competition_distance_scaler, competition_time_month,
        promo_time_week, year e store_type.
        """

        # ── Carregamento dos artefatos de treino ────────────────────────────────
        with open(models_path_parameters / 'rs_distance.pkl',  'rb') as f:
            self.competition_distance_scaler = pickle.load(f)
        with open(models_path_parameters / 'rs_time.pkl',      'rb') as f:
            self.competition_time_month      = pickle.load(f)
        with open(models_path_parameters / 'mms_promo.pkl',    'rb') as f:
            self.promo_time_week             = pickle.load(f)
        with open(models_path_parameters / 'mms_year.pkl',     'rb') as f:
            self.year                        = pickle.load(f)
        with open(models_path_parameters / 'store_type.pkl',   'rb') as f:
            self.store_type                  = pickle.load(f)

    #Função 2: Limpeza dos Dados Recebidos
    def data_cleaning(self, df1):
        """
        Renomeia as colunas do payload recebido para snake_case e preenche
        os valores ausentes com o mesmo critério usado no treino do modelo.

        Etapas aplicadas
        ----------------
        1. Seleção apenas das colunas esperadas (cols_old) e renomeação
           para snake_case via inflection.underscore.
        2. 'open' ausente vira 1 — convenção do Kaggle de que Open=NaN
           significa loja aberta.
        3. Conversão de 'date' para datetime.
        4. 'competition_distance' ausente vira 200000.0 (loja sem
           concorrente próximo conhecido).
        5. 'competition_open_since_month'/'_year' ausentes assumem o
           mês/ano da própria data do pedido.
        6. 'promo2_since_week'/'_year' ausentes assumem a semana/ano da
           própria data do pedido.
        7. Cálculo de 'is_promo' a partir de 'promo_interval' e do mês da
           data.

        Parâmetros
        ----------
        df1 : pd.DataFrame
            DataFrame bruto recebido pela API, já validado por
            handler._validate().

        Retorna
        -------
        df1 : pd.DataFrame
            DataFrame com colunas renomeadas em snake_case e valores
            ausentes preenchidos, pronto para feature_engineering().
        """

        # ── 1. Seleção e renomeação das colunas ───────────────────────────────────
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
                    'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

        # Select only expected columns before renaming to guard against extra payload columns
        df1 = df1[cols_old].copy()

        snakecase = lambda x: inflection.underscore(x)
        cols_new = list(map(snakecase, cols_old))
        df1.columns = cols_new

        # ── 2. Preenchimento de 'open' ausente ─────────────────────────────────────
        # Kaggle convention: Open=NaN means the store was open; make it explicit
        df1['open'] = df1['open'].fillna(1).astype(int)

        # ── 3. Conversão da data ────────────────────────────────────────────────────
        df1['date'] = pd.to_datetime(df1['date'])

        # ── 4. Preenchimento de 'competition_distance' ausente ─────────────────────
        # pd.isna handles both float NaN and Python None (JSON null), unlike math.isnan
        df1['competition_distance'] = df1['competition_distance'].apply(
            lambda x: 200000.0 if pd.isna(x) else x)

        # ── 5. Preenchimento de competition_open_since_month/year ausentes ─────────
        df1['competition_open_since_month'] = df1.apply(
            lambda x: x['date'].month if pd.isna(x['competition_open_since_month'])
                      else x['competition_open_since_month'], axis=1)

        df1['competition_open_since_year'] = df1.apply(
            lambda x: x['date'].year if pd.isna(x['competition_open_since_year'])
                      else x['competition_open_since_year'], axis=1)

        # ── 6. Preenchimento de promo2_since_week/year ausentes ────────────────────
        # int() cast avoids UInt32 dtype from isocalendar() mixing with float in the column
        df1['promo2_since_week'] = df1.apply(
            lambda x: int(x['date'].isocalendar().week) if pd.isna(x['promo2_since_week'])
                      else x['promo2_since_week'], axis=1)

        df1['promo2_since_year'] = df1.apply(
            lambda x: x['date'].year if pd.isna(x['promo2_since_year'])
                      else x['promo2_since_year'], axis=1)

        # ── 7. Cálculo de is_promo ──────────────────────────────────────────────────
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

    #Função 3: Engenharia de Atributos
    def feature_engineering(self, df2):
        """
        Deriva as variáveis de data, tempo de concorrência/promoção e
        remapeia 'assortment'/'state_holiday' para os rótulos usados no
        treino do modelo.

        Etapas aplicadas
        ----------------
        1. Extração de year, month, day, week_of_year e year_week a
           partir de 'date'.
        2. Cálculo de 'competition_time_month': meses desde o início da
           concorrência.
        3. Cálculo de 'promo_time_week': semanas desde o início da
           promoção estendida (promo2).
        4. Remapeamento de 'assortment' (a/b/c) para basic/extra/extended.
        5. Remapeamento de 'state_holiday' (a/b/c/0) para os rótulos de
           feriado usados no treino.
        6. Remoção dos dias com loja fechada (open == 0) — não há venda a
           prever nesses dias.
        7. Descarte das colunas auxiliares que não entram no modelo.

        Parâmetros
        ----------
        df2 : pd.DataFrame
            Saída de data_cleaning().

        Retorna
        -------
        df2 : pd.DataFrame
            DataFrame só com os dias em aberto e as variáveis derivadas,
            pronto para data_preparation(). Pode vir vazio quando todos os
            dias do payload estão fechados.
        """

        df2 = df2.copy()

        # ── 1. Componentes de data ──────────────────────────────────────────────────
        df2['year']         = df2['date'].dt.year
        df2['month']        = df2['date'].dt.month
        df2['day']          = df2['date'].dt.day
        # astype(int) converts UInt32 returned by isocalendar() to plain int64
        df2['week_of_year'] = df2['date'].dt.isocalendar().week.astype(int)
        df2['year_week']    = df2['date'].dt.strftime('%Y-%W')

        # ── 2. Tempo de concorrência em meses ───────────────────────────────────────
        df2['competition_since'] = df2.apply(
            lambda x: datetime.datetime(year=x['competition_open_since_year'],
                                        month=x['competition_open_since_month'], day=1), axis=1)
        # .dt.days extracts integer days from the Timedelta Series before dividing,
        # avoiding the float result that broke .days attribute access in the original
        df2['competition_time_month'] = (df2['date'] - df2['competition_since']).dt.days // 30

        # ── 3. Tempo de promoção estendida em semanas ───────────────────────────────
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply(
            lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))
        df2['promo_time_week'] = (df2['date'] - df2['promo_since']).dt.days // 7

        # ── 4. Remapeamento de assortment ───────────────────────────────────────────
        df2['assortment'] = df2['assortment'].apply(
            lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

        # ── 5. Remapeamento de state_holiday ────────────────────────────────────────
        df2['state_holiday'] = df2['state_holiday'].apply(
            lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b'
                      else 'christmas' if x == 'c' else 'regular_day')

        # ── 6. Remoção dos dias com loja fechada ────────────────────────────────────
        df2 = df2[df2['open'] != 0]

        # ── 7. Descarte das colunas auxiliares ──────────────────────────────────────
        cols_drop = ['open', 'promo_interval', 'month_map', 'is_promo',
                     'year_week', 'competition_since', 'promo_since']
        df2 = df2.drop(cols_drop, axis=1)

        return df2

    #Função 4: Preparação Final das Features
    def data_preparation(self, df5):
        """
        Aplica os scalers/encoders treinados e o encoding cíclico
        (seno/cosseno) às variáveis temporais, produzindo a matriz de
        features na ordem exata esperada pelo modelo.

        Etapas aplicadas
        ----------------
        1. RobustScaler em 'competition_distance' e
           'competition_time_month'; MinMaxScaler em 'promo_time_week' e
           'year' — todos ajustados no notebook de treino.
        2. One-hot encoding de 'state_holiday', garantindo as 4 colunas
           mesmo quando alguma categoria não aparece neste lote.
        3. LabelEncoder em 'store_type' e mapeamento manual de
           'assortment' para 1/2/3.
        4. Encoding cíclico (seno/cosseno) de day_of_week, month, day e
           week_of_year.
        5. Seleção e ordenação final das colunas em cols_selected.

        Parâmetros
        ----------
        df5 : pd.DataFrame
            Saída de feature_engineering().

        Retorna
        -------
        X_test : pd.DataFrame
            Matriz de features numérica, na ordem exata usada no treino
            do XGBoost.
        """

        # API context: no train/test split needed; apply transforms directly to the full input
        X_test = df5.copy()

        # ── 1. Scalers ajustados no treino ──────────────────────────────────────────
        X_test['competition_distance']   = self.competition_distance_scaler.transform(X_test[['competition_distance']].values)
        X_test['competition_time_month'] = self.competition_time_month.transform(X_test[['competition_time_month']].values)
        X_test['promo_time_week']        = self.promo_time_week.transform(X_test[['promo_time_week']].values)
        X_test['year']                   = self.year.transform(X_test[['year']].values)

        # ── 2. One-hot encoding de state_holiday ────────────────────────────────────
        X_test = pd.get_dummies(X_test, prefix=['state_holiday'], columns=['state_holiday'], dtype=int)
        # Guarantee all OHE columns exist regardless of which categories appear in this batch
        for col in ['state_holiday_christmas', 'state_holiday_easter_holiday',
                    'state_holiday_public_holiday', 'state_holiday_regular_day']:
            if col not in X_test.columns:
                X_test[col] = 0

        # ── 3. Encoding de store_type e assortment ──────────────────────────────────
        X_test['store_type'] = self.store_type.transform(X_test['store_type'])

        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        X_test['assortment'] = X_test['assortment'].map(assortment_dict)

        # ── 4. Encoding cíclico das variáveis temporais ─────────────────────────────
        # Vectorized cyclic encoding (no per-row apply overhead)
        X_test['day_of_week_sin']  = np.sin(X_test['day_of_week']  * (2. * np.pi / 7))
        X_test['day_of_week_cos']  = np.cos(X_test['day_of_week']  * (2. * np.pi / 7))
        X_test['month_sin']        = np.sin(X_test['month']         * (2. * np.pi / 12))
        X_test['month_cos']        = np.cos(X_test['month']         * (2. * np.pi / 12))
        X_test['day_sin']          = np.sin(X_test['day']           * (2. * np.pi / 30))
        X_test['day_cos']          = np.cos(X_test['day']           * (2. * np.pi / 30))
        X_test['week_of_year_sin'] = np.sin(X_test['week_of_year']  * (2. * np.pi / 52))
        X_test['week_of_year_cos'] = np.cos(X_test['week_of_year']  * (2. * np.pi / 52))

        # ── 5. Seleção e ordenação final das colunas ────────────────────────────────
        cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance',
                         'competition_open_since_month', 'competition_open_since_year',
                         'promo2', 'promo2_since_week', 'promo2_since_year',
                         'competition_time_month', 'promo_time_week',
                         'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
                         'day_sin', 'day_cos', 'week_of_year_sin', 'week_of_year_cos']

        return X_test[cols_selected]

    #Função 5: Predição de Vendas
    def get_prediction(self, model, original_data, test_data):
        """
        Roda o modelo treinado sobre as features preparadas e devolve a
        previsão de cada dia no formato JSON consumido pela API e pelo
        bot do Telegram.

        Responde às perguntas:
        "Quanto a loja vai vender em cada um dos próximos dias?"

        Parâmetros
        ----------
        model : xgboost.XGBRegressor
            Modelo XGBoost tunado, carregado do pickle em handler.py.
        original_data : pd.DataFrame
            DataFrame original recebido pela API (test_raw), antes da
            limpeza — usado para devolver Store/Date no formato original
            ao usuário.
        test_data : pd.DataFrame
            Saída de data_preparation(), com o mesmo índice de
            original_data nas linhas que sobraram após o filtro de lojas
            fechadas.

        Retorna
        -------
        str
            JSON (orient='records') com uma lista de
            {Store, Date, prediction} — prediction já revertido do log1p
            usado no treino via np.expm1.
        """

        # ── 1. Predição sobre as features preparadas ────────────────────────────────
        pred = model.predict(test_data)

        # ── 2. Alinhamento de índice e reversão do log1p ────────────────────────────
        # Index alignment: feature_engineering may drop open=0 rows, so len(pred) <= len(original_data)
        original_data = original_data.copy()
        original_data.loc[test_data.index, 'prediction'] = np.expm1(pred)

        return original_data[['Store', 'Date', 'prediction']].to_json(orient='records', date_format='iso')
