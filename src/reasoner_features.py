from owlready2 import *
import datetime
import os

ONTO_FILENAME = "movie_rating_ontology.owl"
ONTO_PATH = os.path.join("ontology", ONTO_FILENAME)

onto = get_ontology(ONTO_PATH).load()

### Controlla e segna nella KB i film d'autore (definiti come avere valutazione media del regista > 7, regista con meno di 8 film, e budget < 20 milioni)
def checkAuteurProjects():
    print("Applicando regola: Auteur Projects")
    for film in onto.Film.instances():
        film.isAuteurProject = False
        if film.hasDirector:
            # Ottengo le caratteristiche dalla KB se sono presenti, altrimenti le imposto ad un valore di default
            avgRating = film.hasDirector[0].avgRatingOfDirectedFilms if film.hasDirector[0].avgRatingOfDirectedFilms else 0.0
            nFilm = film.hasDirector[0].nFilmsDirected if film.hasDirector[0].nFilmsDirected else float('inf')
            budget = film.budget if film.budget else float('inf')
            if avgRating > 7.0 and nFilm < 8 and budget < 20000000:
                film.isAuteurProject = True

### Controlla e segna nella KB i film di prestigio, se il regista o un attore ha vinto un premio, è stato rilasciato nella stagione dei premi (da ottobre a dicembre) e ha un budget > 15 milioni
def checkPrestigeProjects():
    print("Applicando regola: Prestige Projects")
    for film in onto.Film.instances():
        film.isPrestigeProject = False
        director = film.hasDirector[0] if film.hasDirector else None
        actors = list(film.hasActor)
        budget = film.budget if film.budget else 0
        releaseMonth = film.releaseDate.month if film.releaseDate else None

        if not director or not releaseMonth:
            continue

        isAwardsSeason = releaseMonth in [10, 11, 12] # il film deve essere rilasciato da ottobre a dicembre
        directorHasAward = director.nAwards > 0 if director.nAwards else False
        actorHasAward = False
        for actor in actors:
            if actor.nAwards and actor.nAwards > 0:
                actorHasAward = True
                break

        if (directorHasAward or actorHasAward) and isAwardsSeason and budget > 15000000:
            film.isPrestigeProject = True

### Controlla e segna nella KB i film con collaborazioni comprovate, se il regista e un attore hanno lavorato insieme ad altri film precedentemente e la valutazione media di tali film è > 7.5
def checkProvenCollaborations():
    print("Applicando regola: Proven Collaborations")
    for film in onto.Film.instances():
        film.hasProvenCollaboration = False
        director = film.hasDirector[0] if film.hasDirector else None
        actors = list(film.hasActor)
        releaseDate = film.releaseDate if film.releaseDate else None

        if not director or not actors or not releaseDate:
            continue

        for actor in actors:
            prevCollaborationRatings = []

            commonFilms = set(director.hasDirected).intersection(set(actor.hasActed))
            for prevFilm in commonFilms: # Ottengo le valutazioni di tutti i film in cui regista e questo attore hanno collaborato
                if prevFilm != film and prevFilm.releaseDate and prevFilm.releaseDate < film.releaseDate and prevFilm.tmdbRating:
                    prevCollaborationRatings.append(prevFilm.tmdbRating)

            # Se la media di tali valutazioni è superiore a 7.5 allora il film ha una collaborazione comprovata
            if prevCollaborationRatings and (sum(prevCollaborationRatings) / len(prevCollaborationRatings)) > 7.5:
                film.hasProvenCollaboration = True
                break     

### Controlla e segna nella KB le persone all'apice della propria carriera, se sono tra il decimo e il ventesimo anno di carriera e la valutazione media dei loro film è > 7.5
def checkCareerPeak():
    print("Applicando regola: Career Peak")
    for person in onto.Person.instances():
        person.personInCareerPeak = False

        if person.careerStartYear:
            careerLength = datetime.date.today().year - person.careerStartYear
            avgRating = 0.0
            if isinstance(person, onto.Director) and person.avgRatingOfDirectedFilms:
                avgRating = person.avgRatingOfDirectedFilms
            elif isinstance(person, onto.Actor) and person.avgRatingOfActedFilms:
                avgRating = person.avgRatingOfActedFilms

            if 10 <= careerLength <= 20 and avgRating > 7.0:
                person.personInCareerPeak = True

### --- ###

print("Inizio applicazione regole di ragionamento")
checkAuteurProjects()
checkPrestigeProjects()
checkProvenCollaborations()
checkCareerPeak()
print("Fine applicazione regole di ragionamento")
onto.save(file = ONTO_PATH, format = "rdfxml")
print(f"KB con feature di ragionamento salvata")       
