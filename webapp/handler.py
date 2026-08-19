#==================================
# Import Library
#==================================

import json
import logging
import pickle
import pandas as pd
import os
from pathlib import Path
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann

#==================================
# Configuration
#==================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_here = Path(__file__).resolve().parent
models_path_ml = _here / 'model'

with open(models_path_ml / 'model_xgb_tunned.pkl', 'rb') as f:
    model = pickle.load(f)
# Single instance at module level: avoids reloading 5 pickle files on every request
pipeline = Rossmann()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB hard limit; Flask returns 413 automatically

REQUIRED_FIELDS = [
    'Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
    'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval'
]
_VALID_STORE_TYPES = {'a', 'b', 'c', 'd'}
_VALID_ASSORTMENTS = {'a', 'b', 'c'}

#===================================
# Functions
#===================================

#Função 1: Validação do Payload Recebido
def _validate(df):
    """
    Confere se o DataFrame recebido tem as colunas e os valores que o
    pipeline de limpeza espera, antes de gastar tempo rodando o modelo
    sobre um payload inválido.

    Etapas de validação
    --------------------
    1. Presença de todas as colunas listadas em REQUIRED_FIELDS.
    2. Coluna 'Date' conversível para datetime.
    3. Valores de 'StoreType' dentro de {'a', 'b', 'c', 'd'}.
    4. Valores de 'Assortment' dentro de {'a', 'b', 'c'}.
    5. 'CompetitionOpenSinceMonth' entre 1 e 12.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame construído a partir do JSON recebido em
        /rossmann/predict, antes de qualquer limpeza.

    Retorna
    -------
    validation_error : dict ou None
        Dicionário com a chave 'error' (e 'fields' quando faltam colunas)
        descrevendo o primeiro problema encontrado, ou None quando o
        payload passa em todas as validações.
    """

    # ── 1. Colunas obrigatórias ───────────────────────────────────────────────
    missing = [f for f in REQUIRED_FIELDS if f not in df.columns]
    if missing:
        return {'error': 'Missing required fields', 'fields': missing}

    # ── 2. Formato da data ────────────────────────────────────────────────────
    try:
        pd.to_datetime(df['Date'])
    except Exception:
        return {'error': "Invalid 'Date' format. Expected YYYY-MM-DD."}

    # ── 3. Valores válidos de StoreType ───────────────────────────────────────
    bad_types = df['StoreType'].dropna()
    bad_types = bad_types[~bad_types.isin(_VALID_STORE_TYPES)]
    if not bad_types.empty:
        return {'error': f"Invalid StoreType: {bad_types.unique().tolist()}. Expected one of {sorted(_VALID_STORE_TYPES)}."}

    # ── 4. Valores válidos de Assortment ──────────────────────────────────────
    bad_assort = df['Assortment'].dropna()
    bad_assort = bad_assort[~bad_assort.isin(_VALID_ASSORTMENTS)]
    if not bad_assort.empty:
        return {'error': f"Invalid Assortment: {bad_assort.unique().tolist()}. Expected one of {sorted(_VALID_ASSORTMENTS)}."}

    # ── 5. Faixa válida de CompetitionOpenSinceMonth ──────────────────────────
    month_col = pd.to_numeric(df['CompetitionOpenSinceMonth'], errors='coerce').dropna()
    bad_months = month_col[(month_col < 1) | (month_col > 12)]
    if not bad_months.empty:
        return {'error': f"CompetitionOpenSinceMonth must be 1-12; got {bad_months.unique().tolist()}."}

    return None


#Função 2: Endpoint de Predição de Vendas
@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    """
    Recebe o payload de um ou mais dias de loja e devolve a previsão de
    vendas de cada um, rodando o pipeline completo de limpeza, feature
    engineering, preparação e o modelo XGBoost tunado.

    Responde às perguntas:
    "Quanto cada loja vai vender nas próximas 6 semanas?"

    Parâmetros
    ----------
    Nenhum — o corpo da requisição chega pelo request do Flask, como um
    objeto JSON (uma loja/dia) ou uma lista de objetos.

    Retorna
    -------
    flask.Response
        JSON com uma lista de {Store, Date, prediction} e status 200 em
        caso de sucesso; '[]' com status 200 quando o payload vem vazio ou
        nenhum dia sobra em aberto após a limpeza; um objeto
        {'error': ...} com status 400 quando o payload é inválido; e
        {'error': 'Internal server error.'} com status 500 para qualquer
        exceção não tratada durante o pipeline.
    """

    # ── 1. Leitura do corpo da requisição ─────────────────────────────────────
    test_json = request.get_json(force=True)

    if test_json is None or test_json == [] or test_json == {}:
        return Response('[]', status=200, mimetype='application/json')

    try:
        # ── 2. Normalização do payload em DataFrame ────────────────────────────
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        elif isinstance(test_json, list):
            if not all(isinstance(item, dict) for item in test_json):
                return Response(
                    json.dumps({'error': 'Each element in the JSON array must be an object.'}),
                    status=400, mimetype='application/json')
            test_raw = pd.DataFrame(test_json)
        else:
            return Response(
                json.dumps({'error': 'Request body must be a JSON object or array of objects.'}),
                status=400, mimetype='application/json')

        # ── 3. Validação do payload ─────────────────────────────────────────────
        validation_error = _validate(test_raw)
        if validation_error:
            return Response(json.dumps(validation_error), status=400, mimetype='application/json')

        # ── 4. Pipeline de limpeza, feature engineering e preparação ───────────
        df1 = pipeline.data_cleaning(test_raw)
        df2 = pipeline.feature_engineering(df1)

        if df2.empty:
            return Response('[]', status=200, mimetype='application/json')

        df3 = pipeline.data_preparation(df2)

        # ── 5. Predição e resposta ──────────────────────────────────────────────
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return Response(df_response, status=200, mimetype='application/json')

    except Exception as e:
        logger.exception('Unhandled error in /rossmann/predict')
        return Response(
            json.dumps({'error': 'Internal server error.'}),
            status=500, mimetype='application/json')

if __name__ == '__main__':
    # For production use: gunicorn -w 4 -b 0.0.0.0:5000 handler:app
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
