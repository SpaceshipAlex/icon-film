from owlready2 import *
import datetime
import os
import numpy as np
import math

ONTO_FILENAME = "movie_rating_ontology.owl"
ONTO_PATH = os.path.join("ontology", ONTO_FILENAME)
# Pesi delle similarità tra due film, inseriti intuitivamente, da fare tuning
SIMILARITY_WEIGHTS = {
    'directorIdentity': 0.15,
    'directorExperience': 0.1,
    'directorAvgRating': 0.1,
    'directorNationality': 0.02,
    'directorCareerPeak': 0.03,
    'actorJaccard': 0.1,
    'actorAvgExperience': 0.05,
    'actorAvgRating': 0.05,
    'genreJaccard': 0.15,
    'budget': 0.1,
    'releaseYear': 0.05,
    'auteurProject': 0.01,
    'prestigeProject': 0.02,
    'provenChemistry': 0.02,
    'studioJaccard': 0.05
}

onto = get_ontology(ONTO_PATH).load()

# Calcola una misura di similarità (LDS) che vale 1 se i due valori sono uguali e 0 se la differenza tra i due valori è >=maxDiff, decrescendo linearmente per i valori intermedi
def getLinearDecreasingSimilarity(firstValue, secondValue, maxDiff):
    return max(0.0, 1.0 - (abs(firstValue - secondValue)) / maxDiff)

# Calcola la similarità tra i registi di due film, che vale 1 se è lo stesso e 0 altrimenti
def getDirectorIdentitySimilarity(firstFilmIRI, secondFilmIRI):
    firstFilm = onto.search_one(iri = f"*{firstFilmIRI.split('#')[1]}")
    secondFilm = onto.search_one(iri = f"*{secondFilmIRI.split('#')[1]}")

    if not firstFilm or not secondFilm or not firstFilm.hasDirector or not secondFilm.hasDirector:
        return 0.0
    
    return float(firstFilm.hasDirector[0] == secondFilm.hasDirector[0])

# Calcola la similarità tra le esperienze dei registi di due film, LDS con maxDiff 20
def getDirectorExperienceSimilarity(firstFilmIRI, secondFilmIRI):
    firstFilm = onto.search_one(iri = f"*{firstFilmIRI.split('#')[1]}")
    secondFilm = onto.search_one(iri = f"*{secondFilmIRI.split('#')[1]}")

    if not firstFilm or not secondFilm or not firstFilm.directorExperienceAtRelease or not secondFilm.directorExperienceAtRelease:
        return 0.0
    
    return getLinearDecreasingSimilarity(firstFilm.directorExperienceAtRelease, secondFilm.directorExperienceAtRelease, 20)

# Calcola la similarità tra le valutazioni medie dei registi di due film, LDS con maxDiff 3
def getDirectorRatingSimilarity(firstFilmIRI, secondFilmIRI):
    firstFilm = onto.search_one(iri = f"*{firstFilmIRI.split('#')[1]}")
    secondFilm = onto.search_one(iri = f"*{secondFilmIRI.split('#')[1]}")

    if not firstFilm or not secondFilm or not firstFilm.directorAvgRatingBeforeFilm or not secondFilm.directorAvgRatingBeforeFilm:
        return 0.0
    
    return getLinearDecreasingSimilarity(firstFilm.directorAvgRatingBeforeFilm, secondFilm.directorAvgRatingBeforeFilm, 3)

# Calcola la similarità tra le nazionalità dei due registi, 1 se la stessa e 0 altrimenti
def getDirectorNationalitySimilarity(firstFilmIRI, secondFilmIRI):
    firstFilm = onto.search_one(iri = f"*{firstFilmIRI.split('#')[1]}")
    secondFilm = onto.search_one(iri = f"*{secondFilmIRI.split('#')[1]}")

    if not firstFilm or not secondFilm or not firstFilm.hasDirector or not secondFilm.hasDirector \
        or not firstFilm.hasDirector[0].nationality or not firstFilm.hasDirector[0].nationality:
        return 0.0
    
    return float(firstFilm.hasDirector[0].nationality == secondFilm.hasDirector[0].nationality)

# Calcola la similarità tra i picchi di carriera dei registi, 1 se sono o non sono entrambi in picco e 0 altrimenti
def getDirectorCareerPeakSimilarity(firstFilmIRI, secondFilmIRI):
    firstFilm = onto.search_one(iri = f"*{firstFilmIRI.split('#')[1]}")
    secondFilm = onto.search_one(iri = f"*{secondFilmIRI.split('#')[1]}")

    if not firstFilm or not secondFilm or not firstFilm.hasDirector or not secondFilm.hasDirector \
        or not firstFilm.hasDirector[0].personInCareerPeak or not firstFilm.hasDirector[0].personInCareerPeak:
        return 0.0
    
    return float(firstFilm.hasDirector[0].personInCareerPeak == secondFilm.hasDirector[0].personInCareerPeak \
                 or firstFilm.hasDirector[0].personInCareerPeak != secondFilm.hasDirector[0].personInCareerPeak)

# Calcola la similarità di Jaccard tra gli insiemi degli attori dei due film
def getActorJaccardSimilarity(firstFilmIRI, secondFilmIRI):
    firstFilm = onto.search_one(iri = f"*{firstFilmIRI.split('#')[1]}")
    secondFilm = onto.search_one(iri = f"*{secondFilmIRI.split('#')[1]}")

    if not firstFilm or not secondFilm:
        return 0.0
    
    firstActors = set(a for a in firstFilm.hasActor)
    secondActors = set(a for a in secondFilm.hasActor)

    if not firstActors or not secondActors:
        return 0.0
    
    return len(firstActors.intersection(secondActors)) / len(firstActors.union(secondActors)) if len(firstActors.union(secondActors)) > 0 else 0.0

# Calcola la similarità tra le esperienze medie degli attori di due film, LDS con maxDiff 20
def getActorExperienceSimilarity(firstFilmIRI, secondFilmIRI):
    firstFilm = onto.search_one(iri = f"*{firstFilmIRI.split('#')[1]}")
    secondFilm = onto.search_one(iri = f"*{secondFilmIRI.split('#')[1]}")

    if not firstFilm or not secondFilm or not firstFilm.actorsAvgExperienceAtRelease or not secondFilm.actorsAvgExperienceAtRelease:
        return 0.0
    
    return getLinearDecreasingSimilarity(firstFilm.actorsAvgExperienceAtRelease, secondFilm.actorsAvgExperienceAtRelease, 20)

# Calcola la similarità tra le valutazioni medie degli attori di due film, LDS con maxDiff 3
def getActorRatingSimilarity(firstFilmIRI, secondFilmIRI):
    firstFilm = onto.search_one(iri = f"*{firstFilmIRI.split('#')[1]}")
    secondFilm = onto.search_one(iri = f"*{secondFilmIRI.split('#')[1]}")

    if not firstFilm or not secondFilm or not firstFilm.actorsAvgRatingInPrevious2Years or not secondFilm.actorsAvgRatingInPrevious2Years:
        return 0.0
    
    return getLinearDecreasingSimilarity(firstFilm.actorsAvgRatingInPrevious2Years, secondFilm.actorsAvgRatingInPrevious2Years, 3)

# Calcola la similarità di Jaccard tra gli insiemi dei generi dei due film
def getGenreJaccardSimilarity(firstFilmIRI, secondFilmIRI):
    firstFilm = onto.search_one(iri = f"*{firstFilmIRI.split('#')[1]}")
    secondFilm = onto.search_one(iri = f"*{secondFilmIRI.split('#')[1]}")

    if not firstFilm or not secondFilm:
        return 0.0
    
    firstGenres = set(g for g in firstFilm.hasGenre)
    secondGenres = set(g for g in secondFilm.hasGenre)

    if not firstGenres or not secondGenres:
        return 0.0
    
    return len(firstGenres.intersection(secondGenres)) / len(firstGenres.union(secondGenres)) if len(firstGenres.union(secondGenres)) > 0 else 0.0

# Calcola la similarità tra i budget di due film, LDS di log(budget + 1) con maxDiff 1.5
def getBudgetSimilarity(firstFilmIRI, secondFilmIRI):
    firstFilm = onto.search_one(iri = f"*{firstFilmIRI.split('#')[1]}")
    secondFilm = onto.search_one(iri = f"*{secondFilmIRI.split('#')[1]}")

    if not firstFilm or not secondFilm or not firstFilm.budget or not secondFilm.budget:
        return 0.0
    
    return getLinearDecreasingSimilarity(math.log(firstFilm.budget + 1), math.log(secondFilm.budget + 1), 1.5)

# Calcola la similarità tra gli anni di uscita di due film, LDS con maxDiff 10
def getReleaseYearSimilarity(firstFilmIRI, secondFilmIRI):
    firstFilm = onto.search_one(iri = f"*{firstFilmIRI.split('#')[1]}")
    secondFilm = onto.search_one(iri = f"*{secondFilmIRI.split('#')[1]}")

    if not firstFilm or not secondFilm or not firstFilm.releaseDate or not secondFilm.releaseDate:
        return 0.0
    
    return getLinearDecreasingSimilarity(firstFilm.releaseDate.year, secondFilm.releaseDate.year, 10)

# Calcola la similarità tra due film a seconda se sono film d'autore, 1 se lo sono o non lo sono entrambi e 0 altrimenti
def getAuteurProjectSimilarity(firstFilmIRI, secondFilmIRI):
    firstFilm = onto.search_one(iri = f"*{firstFilmIRI.split('#')[1]}")
    secondFilm = onto.search_one(iri = f"*{secondFilmIRI.split('#')[1]}")

    if not firstFilm or not secondFilm or not firstFilm.isAuteurProject or not secondFilm.isAuteurProject:
        return 0.0
    
    return float(firstFilm.isAuteurProject == secondFilm.isAuteurProject \
                 or firstFilm.isAuteurProject != secondFilm.isAuteurProject)

# Calcola la similarità tra due film a seconda se sono film di prestigio, 1 se lo sono o non lo sono entrambi e 0 altrimenti
def getPrestigeProjectSimilarity(firstFilmIRI, secondFilmIRI):
    firstFilm = onto.search_one(iri = f"*{firstFilmIRI.split('#')[1]}")
    secondFilm = onto.search_one(iri = f"*{secondFilmIRI.split('#')[1]}")

    if not firstFilm or not secondFilm or not firstFilm.isPrestigeProject or not secondFilm.isPrestigeProject:
        return 0.0
    
    return float(firstFilm.isPrestigeProject == secondFilm.isPrestigeProject \
                 or firstFilm.isPrestigeProject != secondFilm.isPrestigeProject)

# Calcola la similarità tra due film a seconda se hanno collaborazioni comprovate, 1 se lo sono o non lo sono entrambi e 0 altrimenti
def getProvenChemistrySimilarity(firstFilmIRI, secondFilmIRI):
    firstFilm = onto.search_one(iri = f"*{firstFilmIRI.split('#')[1]}")
    secondFilm = onto.search_one(iri = f"*{secondFilmIRI.split('#')[1]}")

    if not firstFilm or not secondFilm or not firstFilm.hasProvenCollaboration or not secondFilm.hasProvenCollaboration:
        return 0.0
    
    return float(firstFilm.hasProvenCollaboration == secondFilm.hasProvenCollaboration \
                 or firstFilm.hasProvenCollaboration != secondFilm.hasProvenCollaboration)

# Calcola la similarità di Jaccard tra gli insiemi degli studi di produzione dei due film
def getStudioJaccardSimilarity(firstFilmIRI, secondFilmIRI):
    firstFilm = onto.search_one(iri = f"*{firstFilmIRI.split('#')[1]}")
    secondFilm = onto.search_one(iri = f"*{secondFilmIRI.split('#')[1]}")

    if not firstFilm or not secondFilm:
        return 0.0
    
    firstStudios = set(s for s in firstFilm.producedByStudio)
    secondStudios = set(s for s in secondFilm.producedByStudio)

    if not firstStudios or not secondStudios:
        return 0.0
    
    return len(firstStudios.intersection(secondStudios)) / len(firstStudios.union(secondStudios)) if len(firstStudios.union(secondStudios)) > 0 else 0.0

# Calcola la similarità tra due film con le misure e i pesi definiti in precedenza
def calcFilmSimilarity(firstFilmIRI, secondFilmIRI, weights):
    if firstFilmIRI == secondFilmIRI:
        return 1.0
    
    totalSimilarity = 0.0
    totalSimilarity += getDirectorIdentitySimilarity(firstFilmIRI, secondFilmIRI) * weights.get('directorIdentity', 0)
    totalSimilarity += getDirectorExperienceSimilarity(firstFilmIRI, secondFilmIRI) * weights.get('directorExperience', 0)
    totalSimilarity += getDirectorRatingSimilarity(firstFilmIRI, secondFilmIRI) * weights.get('directorAvgRating', 0)
    totalSimilarity += getDirectorNationalitySimilarity(firstFilmIRI, secondFilmIRI) * weights.get('directorNationality', 0)
    totalSimilarity += getDirectorCareerPeakSimilarity(firstFilmIRI, secondFilmIRI) * weights.get('directorCareerPeak', 0)
    totalSimilarity += getActorJaccardSimilarity(firstFilmIRI, secondFilmIRI) * weights.get('actorJaccard', 0)
    totalSimilarity += getActorExperienceSimilarity(firstFilmIRI, secondFilmIRI) * weights.get('actorAvgExperience', 0)
    totalSimilarity += getActorRatingSimilarity(firstFilmIRI, secondFilmIRI) * weights.get('actorAvgRating', 0)
    totalSimilarity += getGenreJaccardSimilarity(firstFilmIRI, secondFilmIRI) * weights.get('genreJaccard', 0)
    totalSimilarity += getBudgetSimilarity(firstFilmIRI, secondFilmIRI) * weights.get('budget', 0)
    totalSimilarity += getReleaseYearSimilarity(firstFilmIRI, secondFilmIRI) * weights.get('releaseYear', 0)
    totalSimilarity += getAuteurProjectSimilarity(firstFilmIRI, secondFilmIRI) * weights.get('auteurProject', 0)
    totalSimilarity += getPrestigeProjectSimilarity(firstFilmIRI, secondFilmIRI) * weights.get('prestigeProject', 0)
    totalSimilarity += getProvenChemistrySimilarity(firstFilmIRI, secondFilmIRI) * weights.get('provenChemistry', 0)
    totalSimilarity += getStudioJaccardSimilarity(firstFilmIRI, secondFilmIRI) * weights.get('studioJaccard', 0)

    return totalSimilarity

### --- ###

allFilmIris = [f.iri for f in onto.Film.instances()] # carico gli IRI di tutti i film
if not allFilmIris:
    print("Nessun film nella KB, impossibile costruire il kernel.")
    exit()

splitID = int(len(allFilmIris) * 0.8) # ottengo l'indice dove verrà divisa la lista dei film tra training e test
trainFilmIris = allFilmIris[:splitID]
testFilmIris = allFilmIris[splitID:]

if not trainFilmIris or not testFilmIris:
    print("Il training e/o test set sono vuoti dopo lo split, impossibile costruire il kernel.")
    exit()

print(f"Inizio costruzione matrici KTrain e KTest, {len(trainFilmIris)} film di training e {len(testFilmIris)} film di test")

lenTrain = len(trainFilmIris)
KTrain = np.zeros((lenTrain, lenTrain))

for i in range(lenTrain):
    for j in range(i, lenTrain):
        sim = calcFilmSimilarity(trainFilmIris[i], trainFilmIris[j], SIMILARITY_WEIGHTS)
        KTrain[i, j] = sim
        KTrain[j, i] = sim
    print(f"KTrain: {i+1}/{lenTrain} righe completate")

lenTest = len(testFilmIris)
KTest = np.zeros((lenTest, lenTrain))

for i in range(lenTest):
    for j in range(lenTrain):
        sim = calcFilmSimilarity(testFilmIris[i], trainFilmIris[j], SIMILARITY_WEIGHTS)
        KTest[i, j] = sim
    print(f"KTest: {i+1}/{lenTest} righe completate")

np.save("kernel/KTrain.npy", KTrain)
np.save("kernel/KTest.npy", KTest)

with open("kernel/trainFilmIris.txt", "w") as f:
    for iri in trainFilmIris:
        f.write(f"{iri}\n")
with open("kernel/testFilmIris.txt", "w") as f:
    for iri in testFilmIris:
        f.write(f"{iri}\n")

print("\nMatrici Kernel e IRI dei film di training e test salvati.")