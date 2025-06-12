from owlready2 import *
import os

ONTO_FILENAME = "movie_rating_ontology.owl"
ONTO_PATH = os.path.join("ontology", ONTO_FILENAME) # non è ../ontology perché la working directory di Python è già nella cartella superiore
ONTO_IRI = "http://test.org/" + ONTO_FILENAME[:-4] ### rimuovo .owl nell'IRI

if os.path.exists(ONTO_PATH):
    try:
        os.remove(ONTO_PATH)
        print("Rimossa ontologia esistente")
    except OSError as e:
        print(f"Errore nella rimozione dell'ontologia corrente: {e}")

onto = get_ontology(ONTO_IRI)

with onto: # per evitare di scrivere (namespace = onto) in ogni classe, si scrive pass
    # CLASSI
    class Film(Thing): pass
    class Person(Thing): pass
    class Director(Person): pass # sotto-classe di Person
    class Actor(Person): pass
    class Studio(Thing): pass
    class Genre(Thing): pass

    # PROPRIETA'
    class hasDirector(ObjectProperty):
        domain = [Film]
        range = [Director]
        functional = True # considero un solo regista per film, per semplicità
    class hasDirected(ObjectProperty):
        domain = [Director]
        range = [Film]
        inverse_property = hasDirector
    class hasActor(ObjectProperty):
        domain = [Film]
        range = [Actor]
    class hasActed(ObjectProperty):
        domain = [Actor]
        range = [Film]
        inverse_property = hasActor
    class producedByStudio(ObjectProperty):
        domain = [Film]
        range = [Studio]
    class hasGenre(ObjectProperty):
        domain = [Film]
        range = [Genre]

    class filmTitle(DataProperty, FunctionalProperty):
        domain = [Film]
        range = [str]
    class tmdbRating(DataProperty, FunctionalProperty):
        domain = [Film]
        range = [float]
    class budget(DataProperty, FunctionalProperty):
        domain = [Film]
        range = [int]
    class revenue(DataProperty, FunctionalProperty):
        domain = [Film]
        range = [int]
    class releaseDate(DataProperty, FunctionalProperty):
        domain = [Film]
        range = [datetime.date]
    class runtime(DataProperty, FunctionalProperty):
        domain = [Film]
        range = [int]
    class directorExperienceAtRelease(DataProperty, FunctionalProperty):
        domain = [Film]
        range = [int]
    class directorAvgRatingBeforeFilm(DataProperty, FunctionalProperty):
        domain = [Film]
        range = [float]
    class directorFilmCountBeforeFilm(DataProperty, FunctionalProperty):
        domain = [Film]
        range = [int]
    class actorsAvgExperienceAtRelease(DataProperty, FunctionalProperty):
        domain = [Film]
        range = [float]
    class actorsAvgRatingInPrevious2Years(DataProperty, FunctionalProperty):
        domain = [Film]
        range = [float]
    class castSize(DataProperty, FunctionalProperty):
        domain = [Film]
        range = [int]
    class isAuteurProject(DataProperty, FunctionalProperty):
        domain = [Film]
        range = [bool]
        comment = ["Indica se si tratta di un film d'autore, definito come avere valutazione media del regista > 7, regista con meno di 8 film, e budget < 20 milioni"]
    class isPrestigeProject(DataProperty, FunctionalProperty):
        domain = [Film]
        range = [bool]
        comment = ["Il film è di prestigio se il regista o un attore ha vinto un premio, è stato rilasciato nella stagione dei premi (da ottobre a dicembre) e ha un budget > 15 milioni"]
    class hasProvenCollaboration(DataProperty, FunctionalProperty):
        domain = [Film]
        range = [bool]
        comment = ["Un film ha una collaborazione comprovata se il regista e un attore hanno lavorato insieme ad altri film precedentemente e la valutazione media di tali film è > 7.5"]

    class personName(DataProperty, FunctionalProperty):
        domain = [Person]
        range = [str]
    class personTmdbID(DataProperty, FunctionalProperty):
        domain = [Person]
        range = [str]
    class personGender(DataProperty, FunctionalProperty): # 1 Donna, 2 Uomo, 3 Non-binario, 0 Non specificato
        domain = [Person]
        range = [int]
    class birthDate(DataProperty, FunctionalProperty):
        domain = [Person]
        range = [datetime.date]
    class careerStartYear(DataProperty, FunctionalProperty):
        domain = [Person]
        range = [int]
    class nationality(DataProperty, FunctionalProperty):
        domain = [Person]
        range = [str]
    class nAwards(DataProperty, FunctionalProperty):
        domain = [Person]
        range = [int]
    class personInCareerPeak(DataProperty, FunctionalProperty):
        domain = [Person]
        range = [bool]
        comment = ["Una persona è all'apice della propria carriera se è tra il decimo e il ventesimo anno di carriera e la valutazione media dei suoi film è > 7.5"]
    
    class avgRatingOfDirectedFilms(DataProperty, FunctionalProperty):
        domain = [Director]
        range = [float]
    class avgRatingOfActedFilms(DataProperty, FunctionalProperty):
        domain = [Actor]
        range = [float]
    class nFilmsDirected(DataProperty, FunctionalProperty):
        domain = [Director]
        range = [int]
    class nFilmsActed(DataProperty, FunctionalProperty):
        domain = [Actor]
        range = [int]
    
onto.save(file = ONTO_PATH, format = "rdfxml")
print("Struttura dell'ontologia definita e salvata")

