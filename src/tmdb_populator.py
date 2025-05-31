import requests
from owlready2 import *
import time
import datetime

import config # dove si trova la API key

ONTO_FILENAME = "movie_rating_ontology.owl"
ONTO_PATH = os.path.join("ontology", ONTO_FILENAME)
RATING_BANDS = [(0, 3.4), (3.5, 5.4), (5.5, 6.9), (7.0, 8.4), (8.5, 10.0)] # fasce di voti secondo le quali effettuare il popolamento
MOVIES_PER_BAND = 2 # numero di film da prelevare per ognuna delle fasce

onto = get_ontology(ONTO_PATH).load()

# Definisco 3 dizionari, per tracciare gli individui già creati e riutilizzarli, e un insieme di ID di film già processari, per evitare di ri-processarli
createdPersons = {}
createdStudios = {}
createdGenres = {}
fetchedMovieIDs = set()

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
        time.sleep(0.1) # Max 10 richieste al secondo
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Errore API TMDb: {e} per endpoint {endpoint} con parametri {params}")
        return None

# Richiede i dettagli del film all'API e aggiunge tutti i dati necessari all'ontologia  
def fetchAndAddMovie(movieID):
    movieDetails = getTMDBData(f"movie/{movieID}", params = {'append_to_response': 'credits,keywords'})
    if movieDetails:
        print(movieDetails.get('title'))
    
### --- ###

for min_rating, max_rating in RATING_BANDS: # per ognuna delle fasce di voti
    print(f"Ottenendo film nella fascia ({min_rating}, {max_rating})")
    page = 1
    nMovies = 0
    while nMovies < MOVIES_PER_BAND:
        discoverParams = {
            'sort_by': 'popularity.desc', # ordino per popolarità decrescente
            'vote_average.gte': min_rating,
            'vote_average.lte': max_rating,
            'vote_count.gte': 50, # escludo film con pochi voti
            'page': page,
            'language': 'it-IT'
        }
        data = getTMDBData("discover/movie", discoverParams) # ottengo i dati da TMDB

        if not data or not data.get('results'): 
            print(f"Fine dei film trovati nella fascia ({min_rating}, {max_rating}) a pagina {page}\n\n")
            break # esco dal while, film finiti

        for movie in data['results']:
            if nMovies >= MOVIES_PER_BAND:
                break # esco dal while se ho superato il numero di film per fascia

            movieID = movie.get('id')
            if movieID in fetchedMovieIDs:
                continue # se ho già elaborato questo film, vado avanti

            fetchAndAddMovie(movieID) # aggiungo film e relativi individui all'ontologia

            fetchedMovieIDs.add(movieID)
            nMovies += 1
            print(f"Film {movieID} processato, ci sono {nMovies} film in questa fascia")
    
        if page >= data.get('total_pages', page):
            break # esco dal while se sono finite le pagine
        page += 1



