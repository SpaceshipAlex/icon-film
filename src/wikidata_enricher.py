from SPARQLWrapper import SPARQLWrapper, JSON
from owlready2 import *
import time
import datetime
import os
import utils

ONTO_FILENAME = "movie_rating_ontology.owl"
ONTO_PATH = os.path.join("ontology", ONTO_FILENAME)
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

onto = get_ontology(ONTO_PATH).load()

### Esegue una query SPARQL su Wikidata
def executeWikidataQuery(sparqlQuery):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setReturnFormat(JSON)
    sparql.addCustomHttpHeader("User-Agent", "KnowledgeEngineeringUniversityProject/1.0 (a.duca1@studenti.uniba.it)") ### Buona pratica
    sparql.setQuery(sparqlQuery)

    try:
        results = sparql.query().convert()
        time.sleep(0.05)
        return results["results"]["bindings"]
    except Exception as e:
        print(f"Errore query Wikidata: {e}\nErrore sollevato per query:\n{sparqlQuery}")
        return []


### A partire dall'ID TMDb, recupera quello di Wikidata, se disponibile
def getWikidataID(tmdbID):
    externalIDS = utils.getTMDBData(f"person/{tmdbID}/external_ids")

    if not externalIDS:
        print(f"Nessun ID esterno trovato per la persona {tmdbID}")
        return None
    
    wikidataID = externalIDS.get('wikidata_id')

    if not wikidataID:
        print(f"Nessun ID wikidata trovato per la persona {tmdbID}")
        return None
    
    return wikidataID

### --- ###

personsToProcess = list(onto.Director.instances()) + list(onto.Actor.instances())
print(f"Inizio arricchimento per {len(personsToProcess)} persone da Wikidata")

nProcessed = 0
nEnriched = 0

with onto:
    for person in personsToProcess:
        personName = person.personName if person.personName else "N/D"
        personTmdbID = person.personTmdbID if person.personTmdbID else "0"
        personWikidataID = getWikidataID(personTmdbID)

        if not personWikidataID:
            continue

        personDetailsQuery = f"""
            SELECT ?birthDate ?nationalityLabel ?personLabel ?awardCount ?careerStart WHERE {{
                BIND(wd:{personWikidataID} AS ?person) # variabile riferimento veloce alla persona

                OPTIONAL {{ ?person wdt:P569 ?birthDate. }} # date of birth

                OPTIONAL {{
                    ?person wdt:P27 ?nationalityUri. # country of citizenship
                    ?nationalityUri rdfs:label ?nationalityLabel.
                    FILTER(LANG(?nationalityLabel) = "en") # ottengo il nome inglese della nazione
                }}

                OPTIONAL {{
                    SELECT(COUNT(*) AS ?awardCount_int) WHERE {{ # sotto-query per contare i premi ricevuti
                        BIND(wd:{personWikidataID} AS ?personSub) # variabile riferimento veloce alla persona nella sotto-query
                        ?personSub p:P166 ?statement. # ottengo lo statement perché devo considerare anche premi uguali vinti più volte in date diverse
                        ?statement ps:P166 ?awardInstance. # award received
                        OPTIONAL {{ ?statement pq:P585 ?date. }} # considero premi vinti più volte ottenendo anche le date
                        VALUES ?majorAwardType {{ wd:Q19020 wd:Q1011547 wd:Q102427 wd:Q28444913 }} # la variabile può indicare Oscar, Golden Globe, BAFTA o Festival di Cannes
                        ?awardInstance (wdt:P31/wdt:P279*) ?majorAwardType. # cerco premi che siano instance of (P31) o subclass of (P279) uno dei quattro definiti sopra
                    }} 
                }} BIND(COALESCE(?awardCount_int, 0) AS ?awardCount) # assegna awardCount_int ad awardCount o 0 se è null

                OPTIONAL {{ ?person wdt:P2031 ?careerStart }}

                SERVICE wikibase:label {{
                    bd:serviceParam wikibase:language "en".
                    ?person rdfs:label ?personLabel. # ottengo la label della persona in inglese
                }}
            }} LIMIT 1
            """
        wikidataResults = executeWikidataQuery(personDetailsQuery)
        nProcessed += 1

        if not wikidataResults:
            print(f"Nessun dettaglio trovato su Wikidata per {personName}")
            continue

        result = wikidataResults[0]

        wikidataBirthDate = result.get('birthDate', {}).get('value')
        if wikidataBirthDate:
            try:
                parsedBirthDate = datetime.datetime.strptime(wikidataBirthDate.split('T')[0], '%Y-%m-%d').date() # separo la stringa YYYY-MM-DDTHH:MM:SS usando T come separatore, per poi trasformarla in data
                person.birthDate = parsedBirthDate
            except ValueError:
                print(f"Formato data di nascita non riconosciuto da Wikidata: {wikidataBirthDate}")
        
        wikidataNationality = result.get('nationalityLabel', {}).get('value')
        if wikidataNationality:
            person.nationality = wikidataNationality
        
        wikidataAwardCount = int(result.get('awardCount', {}).get('value', 0))
        if wikidataAwardCount >= 0:
            person.nAwards = wikidataAwardCount

        wikidataCareerStart = result.get('careerStart', {}).get('value')
        if wikidataCareerStart:
            person.careerStartYear = int(wikidataCareerStart.split('T')[0].split('-')[0]) # ottengo l'anno dalla data inizio della carriera
        
        nEnriched += 1
        print(f"Arricchito {personName}, totale {nEnriched} persone su {nProcessed}")

onto.save(file = ONTO_PATH, format = "rdfxml")
print(f"\nKnowledge Base arricchita da Wikidata salvata")
print(f"Persone totali in KB: {len(personsToProcess)}")
print(f"Persone processate: {nProcessed}")
print(f"Persone arricchite con dati Wikidata: {nEnriched}")