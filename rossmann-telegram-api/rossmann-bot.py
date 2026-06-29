import os
import requests
import json
import pandas as pd
from pathlib import Path
from flask import Flask, request, Response

if 'main_path' not in globals():
    main_path = Path(__file__).resolve().parent

if 'test_df' not in globals():
    test_df  = main_path / "assets" / "data" / "test.csv"
if 'store_df' not in globals():
    store_df = main_path / "assets" / "data" / "store.csv"

TOKEN = os.environ.get('TELEGRAM_TOKEN')

def send_message(chat_id, text):
    url = 'https://api.telegram.org/bot{}/sendMessage'.format(TOKEN)
    r = requests.post(url, json={'chat_id': chat_id, 'text': text})
    print('Status Code {}'.format(r.status_code))
    return None


def load_dataset (store_id):
    df_test_raw  = pd.read_csv(test_df)
    df_store_raw = pd.read_csv(store_df)

    #merge test dataset + store
    df_test = pd.merge(df_test_raw, df_store_raw, how="left", on="Store")

    #choose store for prediction
    df_test = df_test[df_test["Store"] == store_id]
    
    if not df_test.empty:
        #remove close days
        df_test = df_test[df_test["Open"] != 0]
        df_test = df_test[~df_test["Open"].isnull()]
        df_test = df_test.drop("ID", axis=1, errors="ignore")

        # pandas to_json converts NaN -> null (JSON-compliant), avoiding InvalidJSONError
        data = df_test.to_json(orient='records')
    
    else:
        data = 'error'
    
    return data

def predict(data):
    # API Call
    url = "https://rossmann-store-sales-3eed298e1c78.herokuapp.com/rossmann/predict"

    r = requests.post(url, data=data, headers={"Content-Type": "application/json"})
    print("Status Code {}".format(r.status_code))
    if r.status_code != 200:
        print("Response:", r.text[:500])

    response_data = r.json()
    print("Records returned:", len(response_data))
    print("Columns:", list(response_data[0].keys()) if response_data else "empty")

    d1 = pd.DataFrame(response_data)
    d1.head()
    
    return d1

def parse_message(message):
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']
    
    store_id = store_id.replace('/','')
    
    try:
        store_id = int (store_id)
    
    except ValueError:
        store_id = 'error'
    
    return chat_id, store_id


#API initialize
app = Flask (__name__)
    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.get_json()
        
        chat_id, store_id = parse_message (message)
        
        if store_id != 'error':
            #loading date
            data = load_dataset(store_id)
            
            if data != 'error':
                #prediction
                d1 = predict(data)
                
                #calculation
                store_col = "Store" if "Store" in d1.columns else "store"
                df2 = d1[[store_col, "prediction"]].groupby(store_col).sum().reset_index()

                #send message
                msg = "Store Number {} will sell R${:,.2f} in the next 6 weeks".format(
                        df2[store_col].values[0],
                        df2["prediction"].values[0])
                send_message(chat_id, msg)
                return Response('Ok', status=200)
                
            else:
                send_message(chat_id, 'Store Not Available')
                return Response('Ok', status=200)
            
    
        else:
            send_message(chat_id, 'Store ID is Wrong')
            return Response('Ok', status=200)

    else:
        return '<h1> Rossmann Telegram BOT </h1>'



