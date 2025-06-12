from owlready2 import *
import os
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split

ONTO_FILENAME = "movie_rating_ontology.owl"
ONTO_PATH = os.path.join("ontology", ONTO_FILENAME)
# Pesi delle similarità tra due film, inseriti dopo una valutazione manuale di quali impattassero di più
# e di meno la distribuzione delle similarità tra film
SIMILARITY_WEIGHTS = {
    'directorIdentity': 0.15,
    'directorExperience': 0.10,
    'directorAvgRating': 0.10,
    'directorNationality': 0.02,
    'directorCareerPeak': 0.03,
    'actorJaccard': 0.1,
    'actorAvgExperience': 0.05,
    'actorAvgRating': 0.05,
    'genreJaccard': 0.15,
    'budget': 0.10,
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
def getDirectorIdentitySimilarity(firstFilm, secondFilm):

    if not firstFilm or not secondFilm or not firstFilm.hasDirector or not secondFilm.hasDirector:
        return 0.0
    
    return float(firstFilm.hasDirector[0] == secondFilm.hasDirector[0])

# Calcola la similarità tra le esperienze dei registi di due film, LDS con maxDiff 20
def getDirectorExperienceSimilarity(firstFilm, secondFilm):

    if not firstFilm or not secondFilm or not firstFilm.directorExperienceAtRelease or not secondFilm.directorExperienceAtRelease:
        return 0.0
    
    return getLinearDecreasingSimilarity(firstFilm.directorExperienceAtRelease, secondFilm.directorExperienceAtRelease, 20)

# Calcola la similarità tra le valutazioni medie dei registi di due film, LDS con maxDiff 3
def getDirectorRatingSimilarity(firstFilm, secondFilm):

    if not firstFilm or not secondFilm or not firstFilm.directorAvgRatingBeforeFilm or not secondFilm.directorAvgRatingBeforeFilm:
        return 0.0
    
    return getLinearDecreasingSimilarity(firstFilm.directorAvgRatingBeforeFilm, secondFilm.directorAvgRatingBeforeFilm, 3)

# Calcola la similarità tra le nazionalità dei due registi, 1 se la stessa e 0 altrimenti
def getDirectorNationalitySimilarity(firstFilm, secondFilm):

    if not firstFilm or not secondFilm or not firstFilm.hasDirector or not secondFilm.hasDirector \
        or not firstFilm.hasDirector[0].nationality or not firstFilm.hasDirector[0].nationality:
        return 0.0
    
    return float(firstFilm.hasDirector[0].nationality == secondFilm.hasDirector[0].nationality)

# Calcola la similarità tra i picchi di carriera dei registi, 1 se sono o non sono entrambi in picco e 0 altrimenti
def getDirectorCareerPeakSimilarity(firstFilm, secondFilm):

    if not firstFilm or not secondFilm or not firstFilm.hasDirector or not secondFilm.hasDirector \
        or not firstFilm.hasDirector[0].personInCareerPeak or not firstFilm.hasDirector[0].personInCareerPeak:
        return 0.0
    
    return float(firstFilm.hasDirector[0].personInCareerPeak == secondFilm.hasDirector[0].personInCareerPeak \
                 or firstFilm.hasDirector[0].personInCareerPeak != secondFilm.hasDirector[0].personInCareerPeak)

# Calcola la similarità di Jaccard tra gli insiemi degli attori dei due film
def getActorJaccardSimilarity(firstFilm, secondFilm):

    if not firstFilm or not secondFilm:
        return 0.0
    
    firstActors = set(a for a in firstFilm.hasActor)
    secondActors = set(a for a in secondFilm.hasActor)

    if not firstActors or not secondActors:
        return 0.0
    
    return len(firstActors.intersection(secondActors)) / len(firstActors.union(secondActors)) if len(firstActors.union(secondActors)) > 0 else 0.0

# Calcola la similarità tra le esperienze medie degli attori di due film, LDS con maxDiff 20
def getActorExperienceSimilarity(firstFilm, secondFilm):

    if not firstFilm or not secondFilm or not firstFilm.actorsAvgExperienceAtRelease or not secondFilm.actorsAvgExperienceAtRelease:
        return 0.0
    
    return getLinearDecreasingSimilarity(firstFilm.actorsAvgExperienceAtRelease, secondFilm.actorsAvgExperienceAtRelease, 20)

# Calcola la similarità tra le valutazioni medie degli attori di due film, LDS con maxDiff 3
def getActorRatingSimilarity(firstFilm, secondFilm):

    if not firstFilm or not secondFilm or not firstFilm.actorsAvgRatingInPrevious2Years or not secondFilm.actorsAvgRatingInPrevious2Years:
        return 0.0
    
    return getLinearDecreasingSimilarity(firstFilm.actorsAvgRatingInPrevious2Years, secondFilm.actorsAvgRatingInPrevious2Years, 3)

# Calcola la similarità di Jaccard tra gli insiemi dei generi dei due film
def getGenreJaccardSimilarity(firstFilm, secondFilm):

    if not firstFilm or not secondFilm:
        return 0.0
    
    firstGenres = set(g for g in firstFilm.hasGenre)
    secondGenres = set(g for g in secondFilm.hasGenre)

    if not firstGenres or not secondGenres:
        return 0.0
    
    return len(firstGenres.intersection(secondGenres)) / len(firstGenres.union(secondGenres)) if len(firstGenres.union(secondGenres)) > 0 else 0.0

# Calcola la similarità tra i budget di due film, LDS di log(budget + 1) con maxDiff 1.5
def getBudgetSimilarity(firstFilm, secondFilm):

    if not firstFilm or not secondFilm or not firstFilm.budget or not secondFilm.budget:
        return 0.0
    
    return getLinearDecreasingSimilarity(math.log(firstFilm.budget + 1), math.log(secondFilm.budget + 1), 1.5)

# Calcola la similarità tra gli anni di uscita di due film, LDS con maxDiff 10
def getReleaseYearSimilarity(firstFilm, secondFilm):

    if not firstFilm or not secondFilm or not firstFilm.releaseDate or not secondFilm.releaseDate:
        return 0.0
    
    return getLinearDecreasingSimilarity(firstFilm.releaseDate.year, secondFilm.releaseDate.year, 10)

# Calcola la similarità tra due film a seconda se sono film d'autore, 1 se lo sono o non lo sono entrambi e 0 altrimenti
def getAuteurProjectSimilarity(firstFilm, secondFilm):

    if not firstFilm or not secondFilm or not firstFilm.isAuteurProject or not secondFilm.isAuteurProject:
        return 0.0
    
    return float(firstFilm.isAuteurProject == secondFilm.isAuteurProject \
                 or firstFilm.isAuteurProject != secondFilm.isAuteurProject)

# Calcola la similarità tra due film a seconda se sono film di prestigio, 1 se lo sono o non lo sono entrambi e 0 altrimenti
def getPrestigeProjectSimilarity(firstFilm, secondFilm):

    if not firstFilm or not secondFilm or not firstFilm.isPrestigeProject or not secondFilm.isPrestigeProject:
        return 0.0
    
    return float(firstFilm.isPrestigeProject == secondFilm.isPrestigeProject \
                 or firstFilm.isPrestigeProject != secondFilm.isPrestigeProject)

# Calcola la similarità tra due film a seconda se hanno collaborazioni comprovate, 1 se lo sono o non lo sono entrambi e 0 altrimenti
def getProvenChemistrySimilarity(firstFilm, secondFilm):

    if not firstFilm or not secondFilm or not firstFilm.hasProvenCollaboration or not secondFilm.hasProvenCollaboration:
        return 0.0
    
    return float(firstFilm.hasProvenCollaboration == secondFilm.hasProvenCollaboration \
                 or firstFilm.hasProvenCollaboration != secondFilm.hasProvenCollaboration)

# Calcola la similarità di Jaccard tra gli insiemi degli studi di produzione dei due film
def getStudioJaccardSimilarity(firstFilm, secondFilm):

    if not firstFilm or not secondFilm:
        return 0.0
    
    firstStudios = set(s for s in firstFilm.producedByStudio)
    secondStudios = set(s for s in secondFilm.producedByStudio)

    if not firstStudios or not secondStudios:
        return 0.0
    
    return len(firstStudios.intersection(secondStudios)) / len(firstStudios.union(secondStudios)) if len(firstStudios.union(secondStudios)) > 0 else 0.0

# Calcola la similarità tra due film con le misure e i pesi definiti in precedenza
def calcFilmSimilarity(firstFilm, secondFilm, weights):
    if firstFilm == secondFilm:
        return 1.0
    
    totalSimilarity = 0.0
    totalSimilarity += getDirectorIdentitySimilarity(firstFilm, secondFilm) * weights.get('directorIdentity', 0)
    totalSimilarity += getDirectorExperienceSimilarity(firstFilm, secondFilm) * weights.get('directorExperience', 0)
    totalSimilarity += getDirectorRatingSimilarity(firstFilm, secondFilm) * weights.get('directorAvgRating', 0)
    totalSimilarity += getDirectorNationalitySimilarity(firstFilm, secondFilm) * weights.get('directorNationality', 0)
    totalSimilarity += getDirectorCareerPeakSimilarity(firstFilm, secondFilm) * weights.get('directorCareerPeak', 0)
    totalSimilarity += getActorJaccardSimilarity(firstFilm, secondFilm) * weights.get('actorJaccard', 0)
    totalSimilarity += getActorExperienceSimilarity(firstFilm, secondFilm) * weights.get('actorAvgExperience', 0)
    totalSimilarity += getActorRatingSimilarity(firstFilm, secondFilm) * weights.get('actorAvgRating', 0)
    totalSimilarity += getGenreJaccardSimilarity(firstFilm, secondFilm) * weights.get('genreJaccard', 0)
    totalSimilarity += getBudgetSimilarity(firstFilm, secondFilm) * weights.get('budget', 0)
    totalSimilarity += getReleaseYearSimilarity(firstFilm, secondFilm) * weights.get('releaseYear', 0)
    totalSimilarity += getAuteurProjectSimilarity(firstFilm, secondFilm) * weights.get('auteurProject', 0)
    totalSimilarity += getPrestigeProjectSimilarity(firstFilm, secondFilm) * weights.get('prestigeProject', 0)
    totalSimilarity += getProvenChemistrySimilarity(firstFilm, secondFilm) * weights.get('provenChemistry', 0)
    totalSimilarity += getStudioJaccardSimilarity(firstFilm, secondFilm) * weights.get('studioJaccard', 0)

    return totalSimilarity

# Effettua la divisione tra train e test set utilizzando la stratificazione a seconda della valutazione
# del film, come è stato fatto quando sono stati caricati nella KB
def splitFilmsStratified(allFilms):
    print("Inizio splitting dei film in train e test set")
    filmData = []
    for film in allFilms:
        if film.tmdbRating:
            filmData.append({'iri': film.iri, 'rating': film.tmdbRating}) # ottengo iri e valutazione da ogni film
    
    filmDataFrame = pd.DataFrame(filmData) # ottengo un dataframe dalla lista di dizionari
    print(f"Numero totale di film con rating: {len(filmDataFrame)}")

    ratingBins = [0, 3.5, 5.5, 6.9, 8.5, 10.0] # definisco i bin di valutazione
    ratingLabels = ['0-3.5', '3.5-5.5', '5.5-6.9', '6.9-8.5', '8.5-10.0']

    # assegno ogni film ad uno dei bin
    filmDataFrame['ratingBin'] = pd.cut(filmDataFrame['rating'], bins = ratingBins, labels = ratingLabels, right = False, include_lowest = True)
    if filmDataFrame['ratingBin'].isnull().any():
        print("Alcuni film non sono stati assegnati ad alcun bin, li rimuovo.") 
        filmDataFrame.dropna(subset = ['ratingBin'], inplace = True) # rimuovo i film non appartenenti a nessun bin (caso impossibile se i dati sono corretti)
    if filmDataFrame.empty:
        print("Nessun film rimasto dopo l'assegnazione ai bin") 
        exit() # termino l'esecuzione se non ci sono film dopo l'assegnazione ai bin
    
    print("Stratificazione in corso...")
    try: # effettuo split train/test set usando i bin se possibile, altrimenti con shuffle
        trainDataFrame, testDataFrame = train_test_split(
            filmDataFrame, 
            test_size = 0.2, # il test set è il 20% del totale
            stratify = filmDataFrame['ratingBin'],
            random_state = 3
        )
    except ValueError as e:
        print(f"Errore durante train_test_split stratificato: {e}")
        print("Effettuo split non stratificato...")
        trainDataFrame, testDataFrame = train_test_split(
            filmDataFrame,
            test_size = 0.2,
            random_state = 3,
            shuffle = True
        )
    
    print(F"Split completato, il training set contiene {len(trainDataFrame)} film e il test set {len(testDataFrame)}")

    return trainDataFrame['iri'].tolist(), testDataFrame['iri'].tolist() # restituisco gli IRI dei film di train/test

### --- ###

allFilms = onto.Film.instances() # carico gli tutti i film
if not allFilms:
    print("Nessun film nella KB, impossibile costruire il kernel.")
    exit()

trainIris, testIris = splitFilmsStratified(allFilms) # ottengo gli IRI di film di train e test usando lo split con stratificazione
trainFilms = []
testFilms = []

for iri in trainIris: # ottengo i film dagli IRI di train
    film = onto.search_one(iri = f"*{iri}")
    if film:
        trainFilms.append(film)
for iri in testIris: # ottengo i film dagli IRI di test
    film = onto.search_one(iri = f"*{iri}")
    if film:
        testFilms.append(film)

if not trainFilms or not testFilms:
    print("Il training e/o test set sono vuoti dopo lo split, impossibile costruire il kernel.")
    exit()

print(f"Inizio costruzione matrici KTrain e KTest, {len(testFilms)} film di training e {len(testFilms)} film di test")

lenTrain = len(trainFilms)
KTrain = np.zeros((lenTrain, lenTrain))

for i in range(lenTrain): # calcolo KTrain: similarità tra tutti i film di training
    for j in range(i, lenTrain):
        sim = calcFilmSimilarity(trainFilms[i], trainFilms[j], SIMILARITY_WEIGHTS)
        KTrain[i, j] = sim
        KTrain[j, i] = sim
    print(f"KTrain: {i+1}/{lenTrain} righe completate")

lenTest = len(testFilms)
KTest = np.zeros((lenTest, lenTrain))

for i in range(lenTest): # calcolo KTest: similarità di tutti i film di test con tutti quelli di training
    for j in range(lenTrain):
        sim = calcFilmSimilarity(testFilms[i], trainFilms[j], SIMILARITY_WEIGHTS)
        KTest[i, j] = sim
    print(f"KTest: {i+1}/{lenTest} righe completate")

np.save("kernel/KTrain.npy", KTrain)
np.save("kernel/KTest.npy", KTest)

print(KTrain.shape)
print(KTest.shape)

# scrivo su file di testo gli IRI dei film di train/test
with open("kernel/trainFilmIris.txt", "w") as f:
    for film in trainFilms:
        f.write(f"{film.iri}\n")
with open("kernel/testFilmIris.txt", "w") as f:
    for film in testFilms:
        f.write(f"{film.iri}\n")

print("\nMatrici Kernel e IRI dei film di training e test salvati.")