from owlready2 import *
import datetime
import os
from collections import defaultdict

ONTO_FILENAME = "movie_rating_ontology.owl"
ONTO_PATH = os.path.join("ontology", ONTO_FILENAME)

onto = get_ontology(ONTO_PATH).load()

### Calcola le feature derivate per le persone
def calcPersonFeatures():
    print("Inizio calcolo features per Persone")
    with onto:
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
    with onto:
        for film in onto.Film.instances():
            print(f"Processando film {film.filmTitle}")
        
calcPersonFeatures()
                

            
