import pickle
import pandas as pd
from pathlib import Path
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann

_here = Path(__file__).resolve().parent
models_path_ml = _here.parent / 'models' / 'ml_model'

model    = pickle.load(open(models_path_ml / 'model_xgb_tunned.pkl', 'rb'))
# Single instance at module level: avoids reloading 5 pickle files on every request
pipeline = Rossmann()

app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json(force=True)

    if test_json:
        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        df1 = pipeline.data_cleaning(test_raw)
        df2 = pipeline.feature_engineering(df1)
        df3 = pipeline.data_preparation(df2)
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return Response(df_response, status=200, mimetype='application/json')

    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run('0.0.0.0')
