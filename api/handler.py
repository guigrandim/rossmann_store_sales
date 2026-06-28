import json
import logging
import pickle
import pandas as pd
from pathlib import Path
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_here = Path(__file__).resolve().parent
models_path_ml = _here.parent / 'models' / 'ml_model'

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

def _validate(df):
    missing = [f for f in REQUIRED_FIELDS if f not in df.columns]
    if missing:
        return {'error': 'Missing required fields', 'fields': missing}

    try:
        pd.to_datetime(df['Date'])
    except Exception:
        return {'error': "Invalid 'Date' format. Expected YYYY-MM-DD."}

    bad_types = df['StoreType'].dropna()
    bad_types = bad_types[~bad_types.isin(_VALID_STORE_TYPES)]
    if not bad_types.empty:
        return {'error': f"Invalid StoreType: {bad_types.unique().tolist()}. Expected one of {sorted(_VALID_STORE_TYPES)}."}

    bad_assort = df['Assortment'].dropna()
    bad_assort = bad_assort[~bad_assort.isin(_VALID_ASSORTMENTS)]
    if not bad_assort.empty:
        return {'error': f"Invalid Assortment: {bad_assort.unique().tolist()}. Expected one of {sorted(_VALID_ASSORTMENTS)}."}

    month_col = pd.to_numeric(df['CompetitionOpenSinceMonth'], errors='coerce').dropna()
    bad_months = month_col[(month_col < 1) | (month_col > 12)]
    if not bad_months.empty:
        return {'error': f"CompetitionOpenSinceMonth must be 1-12; got {bad_months.unique().tolist()}."}

    return None

@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json(force=True)

    if test_json is None or test_json == [] or test_json == {}:
        return Response('[]', status=200, mimetype='application/json')

    try:
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

        validation_error = _validate(test_raw)
        if validation_error:
            return Response(json.dumps(validation_error), status=400, mimetype='application/json')

        df1 = pipeline.data_cleaning(test_raw)
        df2 = pipeline.feature_engineering(df1)

        if df2.empty:
            return Response('[]', status=200, mimetype='application/json')

        df3 = pipeline.data_preparation(df2)
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return Response(df_response, status=200, mimetype='application/json')

    except Exception as e:
        logger.exception('Unhandled error in /rossmann/predict')
        return Response(
            json.dumps({'error': 'Internal server error.'}),
            status=500, mimetype='application/json')

if __name__ == '__main__':
    # For production use: gunicorn -w 4 -b 0.0.0.0:5000 handler:app
    app.run('0.0.0.0')
