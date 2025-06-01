from owlready2 import *
import datetime
import utils

ONTO_FILENAME = "movie_rating_ontology.owl"
ONTO_PATH = os.path.join("ontology", ONTO_FILENAME)
RATING_BANDS = [(0, 3.4), (3.5, 5.4), (5.5, 6.9), (7.0, 8.4), (8.5, 10.0)] # fasce di voti secondo le quali effettuare il popolamento
MOVIES_PER_BAND = 3 # numero di film da prelevare per ognuna delle fasce

onto = get_ontology(ONTO_PATH).load()

# Definisco 3 dizionari, per tracciare gli individui già creati e riutilizzarli, e un insieme di ID di film già processari, per evitare di ri-processarli
# Le chiavi di questi dizionari saranno (ID, Tipo) con Tipo che può essere Genre, Actor, Director, Studio, per maggiore chiarezza
createdPersons = {}
createdStudios = {}
createdGenres = {}
fetchedMovieIDs = set()
    
# Sanifica un nome sostituendo tutti i caratteri non alfanumerici (c.isalnum() = false) con un trattino basso, in modo da renderli adatti per l'IRI dell'ontologia
def sanitizeName(name):
    return "".join(c if c.isalnum() else "_" for c in str(name))


# Richiede i dettagli del film all'API e aggiunge tutti i dati necessari alla KB  
def fetchAndAddMovie(movieID):
    movieDetails = utils.getTMDBData(f"movie/{movieID}", params = {'append_to_response': 'credits,keywords'})
    if movieDetails:
        with onto:
            filmTitleSanitized = sanitizeName(movieDetails.get('title'))
            filmIRI = f"Film_{movieDetails.get('id')}_{filmTitleSanitized}"

            # Creo un nuovo film nella KB se non esiste già
            currentFilm = onto.search_one(iri = f"*{filmIRI}")
            if not currentFilm:
                currentFilm = onto.Film(filmIRI)
            currentFilm.filmTitle = movieDetails.get('title')
            currentFilm.tmdbRating = float(movieDetails.get('vote_average', 0.0))
            currentFilm.budget = int(movieDetails.get('budget', 0))
            currentFilm.revenue = int(movieDetails.get('revenue', 0))
            currentFilm.runtime = int(movieDetails.get('runtime', 0))
            if movieDetails.get('release_date'):
                try:
                    currentFilm.releaseDate = datetime.datetime.strptime(movieDetails.get('release_date'), '%Y-%m-%d').date()
                except ValueError:
                    print(f"Formato della data di uscita non valido per {movieDetails.get('title')}")
            
            # Ciclo tra tutti i generi del film
            for genreData in movieDetails.get('genres', []):
                genreName = genreData.get('name')
                genreID = genreData.get('id')
                genreKey = (str(genreID), "Genre")

                # Se il genere non esiste nel dizionario e nella KB, lo creo e lo aggiungo 
                if genreKey not in createdGenres:
                    genreIRI = f"Genre_{genreID}_{sanitizeName(genreName)}"
                    currentGenre = onto.search_one(iri = f"*{genreIRI}")
                    if not currentGenre:
                        currentGenre = onto.Genre(genreIRI)
                        currentGenre.label.append(genreName)
                    createdGenres[genreKey] = currentGenre
                
                currentFilm.hasGenre.append(createdGenres[genreKey])

            # Ottengo il primo tra i registi listati e lo aggiungo al dizionario e alla KB, se non esiste
            directorData = next((d for d in movieDetails.get('credits', {}).get('crew', []) if d.get('job') == 'Director'), None)
            if directorData:
                directorName = directorData.get('name')
                directorID = directorData.get('id')
                directorGender = directorData.get('gender')
                personKey = (str(directorID), "Director")

                if personKey not in createdPersons:
                    directorIRI = f"Director_{directorID}_{sanitizeName(directorName)}"
                    currentDirector = onto.search_one(iri = f"*{directorIRI}")
                    if not currentDirector:
                        currentDirector = onto.Director(directorIRI)
                        currentDirector.personName = directorName
                        currentDirector.personTmdbID = directorID
                        currentDirector.personGender = directorGender
                    createdPersons[personKey] = currentDirector

                currentFilm.hasDirector.append(createdPersons[personKey])

            # Ottengo i primi 5 attori listati e li aggiungo al dizionario e alla KB, se non esistono
            for actorData in movieDetails.get('credits', {}).get('cast', [])[:5]:
                actorName = actorData.get('name')
                actorID = actorData.get('id')
                actorGender = actorData.get('gender')
                personKey = (str(actorID), "Actor")

                if personKey not in createdPersons:
                    actorIRI = f"Actor_{actorID}_{sanitizeName(actorName)}"
                    currentActor = onto.search_one(iri = f"*{actorIRI}")
                    if not currentActor:
                        currentActor = onto.Actor(actorIRI)
                        currentActor.personName = actorName
                        currentActor.personTmdbID = actorID
                        currentActor.personGender = actorGender
                    createdPersons[personKey] = currentActor

                currentFilm.hasActor.append(createdPersons[personKey])

            # Ottengo i primi 2 studi di produzione listati e li aggiungo al dizionario e alla KB, se non esistono
            for companyData in movieDetails.get('production_companies', [])[:2]:
                studioName = companyData.get('name')
                studioID = companyData.get('id')
                studioKey = (str(studioID), "Studio")

                if studioKey not in createdStudios:
                    studioIRI = f"Studio_{studioID}_{sanitizeName(studioName)}"
                    currentStudio = onto.search_one(iri = f"*{studioIRI}")
                    if not currentStudio:
                        currentStudio = onto.Studio(studioIRI)
                        currentStudio.label.append(studioName)
                    createdStudios[studioKey] = currentStudio
                
                currentFilm.producedByStudio.append(createdStudios[studioKey])   
    
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
        data = utils.getTMDBData("discover/movie", discoverParams) # ottengo i dati da TMDB

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

onto.save(file = ONTO_PATH, format = "rdfxml")
print(f"KB popolata salvata")
print(f"Totale film processati: {len(fetchedMovieIDs)}")
print(f"Totale persone create: {len(createdPersons)}")
print(f"Totale generi creati: {len(createdGenres)}")
print(f"Totale studi creati: {len(createdStudios)}")
