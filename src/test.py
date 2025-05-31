import requests
from SPARQLWrapper import SPARQLWrapper, JSON

import config

### API TMDB

headers = { # Da documentazione API
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI4MDRmNjViYzc2MjgwYjdjNTgwOWU3ODMyZGRjMzRkOCIsIm5iZiI6MTY0NTEwOTI2NS4zNjIsInN1YiI6IjYyMGU2MDExZjc5NGFkMDA2YWMwMTRmNSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Ln5tzSqLmjK2wx5d3132fAS9S5AEUlmiX_w38LbpEfI"
}

try:
    response = requests.get(f"{config.TMDB_BASE_URL}/movie/812/external_ids", headers = headers) # Chiedo gli ID esterni di un film (tra cui ID Wikidata)
    response.raise_for_status(); # Solleva eccezione in caso di errori HTTP
    print(f"TMDB:\n{response.text}\n\n")
except requests.exceptions.RequestException as e:
    print(f"Errore API TMDb: {e}")


### API WIKIDATA

sparql_endpoint = "https://query.wikidata.org/sparql"
sparql = SPARQLWrapper(sparql_endpoint)
sparql.setReturnFormat(JSON)
sparql.addCustomHttpHeader("User-Agent", "KnowledgeEngineeringUniversityProject/1.0 (a.duca1@studenti.uniba.it)") ### Buona pratica

# Nella query SPARQL sostituire tutte le { con {{ altrimenti Python restituisce un errore
# Query di esempio, cerca tutti gli instance of (P31) house cat (Q146)
sparql.setQuery(f""" 
    SELECT ?item ?itemLabel WHERE
    {{
     ?item wdt:P31 wd:Q146. 
     SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],mul,en". }}
    }}
    LIMIT 10
    """
)

try:
    results = sparql.query().convert()
    print(f"Wikidata:\n{results["results"]["bindings"]}")
except Exception as e:
    print(f"Errore query Wikidata: {e}")