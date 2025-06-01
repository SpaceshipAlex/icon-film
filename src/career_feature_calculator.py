from owlready2 import *
import datetime
import os

ONTO_FILENAME = "movie_rating_ontology.owl"
ONTO_PATH = os.path.join("ontology", ONTO_FILENAME)

onto = get_ontology(ONTO_PATH).load()

### Calcola le feature derivate per le persone
def calcPersonFeatures():
    print("Inizio calcolo features per Persone")
    for person in onto.Person.instances():
        print(f"Processando persona {person.personName}")
        if isinstance(person, onto.Director):
            filmsDirectedRating = []
            for film in list(person.hasDirected): # relazione inversa di hasDirector, ottiene film di cui è regista
                if film.tmdbRating:
                    filmsDirectedRating.append(film.tmdbRating)
            
            filmsDirectedAvgRating = 0.0
            if filmsDirectedRating:
                filmsDirectedAvgRating = sum(filmsDirectedRating) / len(filmsDirectedRating) # calcolo la valutazione media dei film diretti
            person.avgRatingOfDirectedFilms = filmsDirectedAvgRating

            print(f"Il regista {person.personName} ha diretto {len(filmsDirectedRating)} film (con valutazione) con una valutazione media di {filmsDirectedAvgRating}")

        if isinstance(person, onto.Actor):
            filmsActedRating = []
            for film in list(person.hasActed):
                if film.tmdbRating:
                    filmsActedRating.append(film.tmdbRating)
            
            filmsActedAvgRating = 0.0
            if filmsActedRating:
                filmsActedAvgRating = sum(filmsActedRating) / len(filmsActedRating) # calcolo la valutazione media dei film in cui ha recitato
            person.avgRatingOfActedFilms = filmsActedAvgRating

            print(f"L'attore {person.personName} ha recitato in {len(filmsActedRating)} film (con valutazione) con una valutazione media di {filmsActedAvgRating}")
        
        if not person.careerStartYear: # se non è impostato, calcolo l'anno di inizio carriera sulla base dei film nella KB
            personFilms = [] # una persona può essere sia attore che regista, in tal caso aggiungo entrambe le liste 
            if isinstance(person, onto.Director):
                personFilms.extend(list(person.hasDirected))
            if isinstance(person, onto.Actor):
                personFilms.extend(list(person.hasActed))

            minYear = float('inf')
            for film in set(personFilms): # trasformo la lista di tutti i film in un insieme e la scorro per trovare il primo
                if film.releaseDate:
                    releaseYear = film.releaseDate.year
                    if releaseYear < minYear:
                        minYear = releaseYear
            
            if minYear != float('inf'):
                person.careerStartYear = minYear
                print(f"Impostato anno di inizio carriera per {person.personName} come {minYear}")


### Calcola le feature derivate per i film
def calcFilmFeatures():
    print("Inizio calcolo feature per Film")
    for film in onto.Film.instances():
        print(f"Processando film {film.filmTitle}")

        if film.hasDirector:
            director = film.hasDirector[0]

            if film.releaseDate and director.careerStartYear: # imposto l'esperienza del regista all'uscita del film
                releaseYear = film.releaseDate.year
                film.directorExperienceAtRelease = max(0, releaseYear - director.careerStartYear)
            
            if film.releaseDate: # imposto la valutazione media del regista prima di questo film e quanti film ha diretto prima di questo
                ratingsDirectedBefore = []
                filmCountBefore = 0
                for prevFilm in director.hasDirected:
                    if prevFilm != film and prevFilm.releaseDate and prevFilm.releaseDate < film.releaseDate:
                        filmCountBefore += 1
                        if prevFilm.tmdbRating: 
                            ratingsDirectedBefore.append(prevFilm.tmdbRating)

                if ratingsDirectedBefore:
                    film.directorAvgRatingBeforeFilm = sum(ratingsDirectedBefore) / len(ratingsDirectedBefore)
                film.directorFilmCountBeforeFilm = filmCountBefore

            actors = list(film.hasActor)
            film.castSize = len(actors)
            if actors and film.releaseDate:
                actorsExperiencesAtRelease = []
                actorsRatingsPrevTwoYears = []
                twoYearsBeforeFilm = film.releaseDate - datetime.timedelta(days=2*365) # ottengo la data di due anni prima dell'uscita

                for actor in actors:
                    if actor.careerStartYear: # aggiungo l'esperienza di questo attore alla lista
                        actorsExperiencesAtRelease.append(max(0, film.releaseDate.year - actor.careerStartYear))

                    singleActorRatingsPrevTwoYears = []

                    for prevFilm in actor.hasActed:
                        if prevFilm != film and prevFilm.releaseDate and prevFilm.releaseDate < film.releaseDate and prevFilm.releaseDate >= twoYearsBeforeFilm and prevFilm.tmdbRating:
                            singleActorRatingsPrevTwoYears.append(prevFilm.tmdbRating)
                    
                    if singleActorRatingsPrevTwoYears:
                        actorsRatingsPrevTwoYears.append(sum(singleActorRatingsPrevTwoYears) / len(singleActorRatingsPrevTwoYears))
                
                if actorsExperiencesAtRelease:
                    film.actorsAvgExperienceAtRelease = sum(actorsExperiencesAtRelease) / len(actorsExperiencesAtRelease)
                if actorsRatingsPrevTwoYears:
                    film.actorsAvgRatingsInPrevious2Years = sum(actorsRatingsPrevTwoYears) / len(actorsRatingsPrevTwoYears)
                        
### --- ###
        
calcPersonFeatures()
calcFilmFeatures()   
onto.save(file = ONTO_PATH, format = "rdfxml")
print(f"KB con feature di carriera salvata")       

            
