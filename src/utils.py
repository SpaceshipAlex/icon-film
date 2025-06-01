import requests
import time

import config

# Esegue una query all'API TMDb
def getTMDBData(endpoint, params = None): 
    if params is None:
        params = {}
    headers = { # Da documentazione API
    "accept": "application/json",
    "Authorization": f"Bearer {config.TMDB_API_KEY}"
    }
    try:
        response = requests.get(f"{config.TMDB_BASE_URL}/{endpoint}", params = params, headers = headers)
        response.raise_for_status()
        time.sleep(0.05) # Max 20 richieste al secondo
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Errore API TMDb: {e} per endpoint {endpoint} con parametri {params}")
        return None