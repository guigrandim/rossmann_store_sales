#==================================
# Import Library
#==================================

import os
import requests
import json
import time
import pandas as pd
from pathlib import Path
from flask import Flask, request, Response

#==================================
# Paths e Configuração
#==================================

if 'main_path' not in globals():
    main_path = Path(__file__).resolve().parent

if 'test_df' not in globals():
    test_df  = main_path / "assets" / "data" / "test.csv"
if 'store_df' not in globals():
    store_df = main_path / "assets" / "data" / "store.csv"

TOKEN = os.environ.get('TELEGRAM_TOKEN')
PREDICT_API_URL = os.environ.get('PREDICT_API_URL', 'https://rossmann-store-sales-3eed298e1c78.herokuapp.com/rossmann/predict')

#===================================
# Functions
#===================================

#Função 1: Envio de Mensagem ao Usuário no Telegram
def send_message(chat_id, text):
    """
    Envia uma mensagem de texto para uma conversa do Telegram via Bot API.

    É o único canal de saída do bot: leva ao usuário tanto a previsão de
    vendas quanto os avisos de erro (loja inexistente, ID inválido, serviço
    de predição ainda hibernando).

    Parâmetros
    ----------
    chat_id : int
        Identificador da conversa do Telegram, extraído por parse_message().
    text : str
        Conteúdo da mensagem a ser enviada.

    Retorna
    -------
    None — a função apenas dispara a requisição e registra no log o status
    code devolvido pela Bot API.
    """

    # ── Chamada à Bot API do Telegram ─────────────────────────────────────────
    url = 'https://api.telegram.org/bot{}/sendMessage'.format(TOKEN)
    r = requests.post(url, json={'chat_id': chat_id, 'text': text})
    print('Status Code {}'.format(r.status_code))
    return None


#Função 2: Carregamento e Preparação dos Dados da Loja
def load_dataset (store_id):
    """
    Monta o payload JSON com os dias em aberto de uma loja específica.

    Junta test.csv (calendário das 6 semanas a prever) com store.csv
    (cadastro da loja), filtra a loja pedida e descarta os dias fechados —
    dia fechado não tem venda a prever e inflaria o total devolvido ao
    usuário.

    Parâmetros
    ----------
    store_id : int
        Número da loja informado pelo usuário no Telegram.

    Retorna
    -------
    data : str
        JSON no formato 'records' com os dias em aberto da loja, pronto para
        ser enviado no corpo da requisição à API de predição. Devolve a
        string 'error' quando o store_id não existe no dataset — o chamador
        precisa testar esse caso antes de seguir para predict().
    """

    # ── 1. Leitura dos datasets ───────────────────────────────────────────────
    df_test_raw  = pd.read_csv(test_df)
    df_store_raw = pd.read_csv(store_df)

    # ── 2. Merge do calendário com o cadastro da loja ─────────────────────────
    df_test = pd.merge(df_test_raw, df_store_raw, how="left", on="Store")

    # ── 3. Filtro da loja pedida ──────────────────────────────────────────────
    df_test = df_test[df_test["Store"] == store_id]

    # ── 4. Remoção dos dias fechados e serialização ───────────────────────────
    if not df_test.empty:
        df_test = df_test[df_test["Open"] != 0]
        df_test = df_test[~df_test["Open"].isnull()]
        df_test = df_test.drop("ID", axis=1, errors="ignore")

        # pandas to_json converts NaN -> null (JSON-compliant), avoiding InvalidJSONError
        data = df_test.to_json(orient='records')

    else:
        data = 'error'

    return data


#Função 3: Consulta à API de Predição de Vendas
def predict(data):
    """
    Envia o payload da loja à API de predição e devolve as previsões diárias.

    O free tier do Render hiberna o serviço de predição após alguns minutos
    sem uso e, enquanto ele acorda, responde com uma página HTML de 429/502
    em vez de JSON. Por isso a chamada é repetida a cada 8 segundos, até seis
    vezes, em vez de quebrar na primeira resposta não-JSON.

    Parâmetros
    ----------
    data : str
        JSON no formato 'records' produzido por load_dataset().

    Retorna
    -------
    d1 : pd.DataFrame
        DataFrame com as colunas Store, Date e prediction — uma linha por dia
        em aberto da loja. Devolve None quando as seis tentativas falham
        (serviço ainda hibernando); o chamador precisa tratar esse caso.
    """

    # ── 1. Chamada à API com retry para o cold start do Render ────────────────
    for attempt in range(6):
        r = requests.post(PREDICT_API_URL, data=data, headers={"Content-Type": "application/json"}, timeout=30)
        print("Status Code {}".format(r.status_code))
        if r.status_code == 200:
            break
        print("Response:", r.text[:500])
        time.sleep(8)
    else:
        return None

    # ── 2. Conversão da resposta em DataFrame ─────────────────────────────────
    response_data = r.json()
    print("Records returned:", len(response_data))
    print("Columns:", list(response_data[0].keys()) if response_data else "empty")

    d1 = pd.DataFrame(response_data)
    d1.head()

    return d1


#Função 4: Leitura do Comando Enviado pelo Usuário
def parse_message(message):
    """
    Extrai o identificador da conversa e o número da loja do update do Telegram.

    O usuário digita o número da loja como um comando (ex: "/22"), então a
    barra é removida antes da conversão para inteiro.

    Parâmetros
    ----------
    message : dict
        Corpo JSON do update enviado pelo webhook do Telegram, contendo as
        chaves message.chat.id e message.text.

    Retorna
    -------
    chat_id : int
        Identificador da conversa, usado para responder ao usuário.
    store_id : int
        Número da loja pedido. Devolve a string 'error' quando o texto não é
        um número — o chamador usa isso para avisar o usuário.
    """

    # ── 1. Extração dos campos do update ──────────────────────────────────────
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']

    # ── 2. Conversão do comando "/NN" em inteiro ──────────────────────────────
    store_id = store_id.replace('/','')

    try:
        store_id = int (store_id)

    except ValueError:
        store_id = 'error'

    return chat_id, store_id


#======================================
# API initialize
#======================================

app = Flask (__name__)

#Função 5: Endpoint do Webhook do Telegram
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Recebe os updates do webhook do Telegram e responde com a previsão da loja.

    Orquestra o fluxo completo: lê o comando, carrega os dados da loja, chama
    a API de predição, soma as previsões diárias em um total de 6 semanas e
    devolve o valor ao usuário. No GET, serve apenas uma página de status.

    Responde às perguntas:
    "Quanto a loja X vai faturar nas próximas 6 semanas?"
    "Essa loja existe na base de previsão?"

    Parâmetros
    ----------
    Nenhum — o corpo do update chega pela request do Flask.

    Retorna
    -------
    Response('Ok', status=200) : flask.Response
        Resposta de todos os caminhos do POST, inclusive os de erro — o
        Telegram reenvia o update quando recebe status diferente de 200,
        então a falha é comunicada ao usuário por mensagem, não pelo status.
    '<h1> Rossmann Telegram BOT </h1>' : str
        Página de status devolvida no GET, útil para verificar se o serviço
        está no ar sem disparar uma previsão.
    """

    if request.method == 'POST':
        message = request.get_json()

        # ── 1. Leitura do comando ─────────────────────────────────────────────
        chat_id, store_id = parse_message (message)

        if store_id != 'error':
            # ── 2. Carregamento dos dados da loja ─────────────────────────────
            data = load_dataset(store_id)

            if data != 'error':
                # ── 3. Predição ───────────────────────────────────────────────
                d1 = predict(data)

                if d1 is None:
                    send_message(chat_id, 'Prediction service is still waking up, please try again in a moment.')
                    return Response('Ok', status=200)

                # ── 4. Soma das previsões diárias em 6 semanas ────────────────
                store_col = "Store" if "Store" in d1.columns else "store"
                df2 = d1[[store_col, "prediction"]].groupby(store_col).sum().reset_index()

                # ── 5. Envio da resposta ao usuário ───────────────────────────
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
