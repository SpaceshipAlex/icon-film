import numpy as np
import os
from owlready2 import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
import pandas as pd 

ONTO_FILENAME = "movie_rating_ontology.owl"
ONTO_PATH = os.path.join("ontology", ONTO_FILENAME)

onto = get_ontology(ONTO_PATH).load()

### Carica gli IRI dei film dal file di training o test
def loadIrisFromFile(filepath):
    if not os.path.exists(filepath):
        print(f"File iri {filepath} inesistente o non trovato")
        return []
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]
    
### Itera sugli IRI dei film e recupera i loro tmdbRating dalla KB
def getTargetsFromKB(iris, qualityThreashold = 7.5):
    targetsRegression = [] # Memorizza le valutazioni dei film per la regressione
    targetsClassification = [] # Memorizza le etichette binarie per la valutazione di un film (alta/bassa qualità) per la classificazione
    validIris = [] # IRI dei film per cui si è trovato un target valido

    for iri in iris:
        film = onto.search_one(iri = f"*{iri.split('#')[1]}")
        if film and film.tmdbRating:
            rating = film.tmdbRating
            targetsRegression.append(rating)
            targetsClassification.append(1 if rating >= qualityThreashold else 0)
            validIris.append(iri)
        else:
            print(f"Rating non trovato per {iri}, sarà escluso dai target.")
    
    return np.array(targetsRegression), np.array(targetsClassification), validIris

### Itera sugli IRI dei film ed estrae le feature di baseline dalla KB, calcolate in career_feature_calculator.py
def getBaselineFeaturesFromKB(iris):
    features = []
    for iri in iris:
        film = onto.search_one(iri = f"*{iri.split('#')[1]}")
        if not film:
            continue

        budget = film.budget if film.budget else 0
        runtime = film.runtime if film.runtime else 0
        releaseYear = film.releaseDate.year if film.releaseDate else 0
        releaseMonth = film.releaseDate.month if film.releaseDate else 0
        directorExp = film.directorExperienceAtRelease if film.directorExperienceAtRelease else 0
        directorRating = film.directorAvgRatingBeforeFilm if film.directorAvgRatingBeforeFilm else 0
        directorFilmsBefore = film.directorFilmCountBeforeFilm if film.directorFilmCountBeforeFilm else 0
        castSize = film.castSize if film.castSize else 0
        actorsExp = film.actorsAvgExperienceAtRelease if film.actorsAvgExperienceAtRelease else 0
        actorsRating = film.actorsAvgRatingInPrevious2Years if film.actorsAvgRatingInPrevious2Years else 0
        isAuteur = int(film.isAuteurProject if film.isAuteurProject else False)
        isPrestige = int(film.isPrestigeProject if film.isPrestigeProject else False)
        hasCollaboration = int(film.hasProvenCollaboration if film.hasProvenCollaboration else False)

        director = film.hasDirector[0] if film.hasDirector else None
        directorTotAwards = (director.nAwards if director.nAwards else 0) if director else 0
        directorTotFilms = (director.nFilmsDirected if director.nFilmsDirected else 0) if director else 0

        features.append([budget, runtime, releaseYear, releaseMonth, directorExp, directorRating, directorFilmsBefore, castSize, 
                         actorsExp, actorsRating, isAuteur, isPrestige, hasCollaboration, directorTotAwards, directorTotFilms])
        
    featureNames = ['budget', 'runtime', 'releaseYear', 'releaseMonth', 'directorExp', 'directorRating', 'directorFilmsBefore', 'castSize',
                    'actorsExp', 'actorsRating', 'isAuteur', 'isPrestige', 'hasCollaborations', 'directorTotAwards', 'directorTotFilms']
    return pd.DataFrame(features, columns = featureNames)
    
### --- ###

trainIris = loadIrisFromFile("kernel/trainFilmIris.txt")
testIris = loadIrisFromFile("kernel/testFilmIris.txt")

if not trainIris or not testIris:
    print("Lista IRI dei film di training e/o test non trovata o vuota, impossibile procedere.")
    exit()

print(f"Caricati {len(trainIris)} IRI di training e {len(testIris)} IRI di test.")

KTrain = None
KTest = None

if os.path.exists("kernel/KTrain.npy") and os.path.exists("kernel/KTest.npy"):
    KTrain = np.load("kernel/KTrain.npy")
    KTest = np.load("kernel/KTest.npy")
    print("Matrici KTrain e KTest caricate.")
else:
    print("Matrici KTrain e KTest non trovate, impossibile procedere.")

trainTargetsReg, trainTargetsClass, trainValidIris = getTargetsFromKB(trainIris)
testTragetsReg, testTargetsClass, testValidIris = getTargetsFromKB(testIris)
trainIndices = None
testIndices = None

# Filtro la matrice KTrain per escludere tutti gli abbinamenti che fanno riferimento a film esclusi dai target
if len(trainValidIris) < len(trainIris):
    trainIndices = [i for i, iri in enumerate(trainIris) if iri in trainValidIris] # Ottengo una lista di indici dei film validi nell'array originale
    if KTrain is not None:
        KTrain = KTrain[np.ix_(trainIndices, trainIndices)] # Filtro KTrain
    trainIris = trainValidIris # Sostituisco la lista originale di IRI con la lista di quelli validi
    print("Filtrata KTrain per gli indici validi")

# Faccio lo stessso per KTest
if len(testValidIris) < len(testIris):
    testIndices = [i for i, iri in enumerate(testIris) if iri in testValidIris]
    if KTest is not None and KTrain is not None and trainIndices is not None:
        KTest = KTest[np.ix_(testIndices, trainIndices)]
    else:
        print("KTrain non è stata filtrata correttamente, Ktest sarà solo filtrata nelle colonne")
        KTest = KTest[testIndices, :]
    testIris = testValidIris
    print("Filtrata KTest per gli indici validi")

print(f"Target di training: {len(trainTargetsReg)} (regressione), {len(trainTargetsClass)} (classificazione)")
print(f"Target di test: {len(testTragetsReg)} (regressione), {len(testTargetsClass)} (classificazione)")

if KTrain is not None and (KTrain.shape[0] != len(trainTargetsReg) or (KTrain.shape[1] != len(trainTargetsReg))):
    print("La dimensione di KTrain è diversa dal numero di target di training dopo il filtraggio, errore.")
    exit()
elif KTest is not None and KTrain is not None and (KTest.shape[0] != len(testTragetsReg) or (KTest.shape[1] != KTrain.shape[0])):
    print("La dimensione di KTest è diversa dal numero di target di training dopo il filtraggio o dal numero di colonne di KTrain, errore.")
    exit()

trainBaselineFeatures = getBaselineFeaturesFromKB(trainIris)
testBaselineFeatures = getBaselineFeaturesFromKB(testIris)
print(f"Feature di baseline caricate: {trainBaselineFeatures.shape} (train), {testBaselineFeatures.shape} (test)")

numericFeatureNames = ['budget', 'runtime', 'releaseYear', 'releaseMonth', 'directorExp', 'directorRating', 'directorFilmsBefore', 'castSize',
                    'actorsExp', 'actorsRating', 'directorTotAwards', 'directorTotFilms'] # Lista di nomi delle feature numeriche, che necessitano di scaling

# Preprocessor: applico il trasformatore StandardScaler, che calcola la media e la deviazione standard di ciascuna colonna numerica e, quando viene usato
# per trasformare i dati, sottrae la media e divide per la deviazione standard. Le colonne non numeriche sono ignorate in questo passo.
baselinePreprocessor = ColumnTransformer (
    transformers = [
        ('num', StandardScaler(), numericFeatureNames) # Applico il trasformatore StandardScaler
    ],
    remainder = 'passthrough' # Ignoro le feature non numeriche (ossia quelle non in numericFeatureNames)
)

# Pipeline per la regressione: concatena il passaggio di preprocessing, dettagliato precedentemente, con il passaggio di regressione. Uso il modello
# RandomForest, con 100 alberi decisionali (valore comune) e random state 3 (in modo da rendere i risultati riproducibili). n_jobs = -1 per usare tutti i core della CPU
rfRegressor = Pipeline(steps = [
    ('preprocessor', baselinePreprocessor),
    ('regressor', RandomForestRegressor(n_estimators = 100, random_state = 3, n_jobs = -1))
])
print("Addestramento Baseline Regressor...")
if not trainBaselineFeatures.empty and len(trainTargetsReg) > 0:
    rfRegressor.fit(trainBaselineFeatures, trainTargetsReg) # Effettuo l'addestramento per la regressione
    print("Baseline Regressor addestrato.")
else:
    print("Dati di training per baseline mancanti o vuoti.")

# Pipeline per la classificazione: come sopra
rfClassifier = Pipeline(steps = [
    ('preprocessor', baselinePreprocessor),
    ('regressor', RandomForestClassifier(n_estimators = 100, random_state = 3, n_jobs = -1))
])
print("Addestramento Baseline Classifier...")
if not trainBaselineFeatures.empty and len(trainTargetsClass) > 0:
    rfRegressor.fit(trainBaselineFeatures, trainTargetsClass) # Effettuo l'addestramento per la classificazione
    print("Baseline Classifier addestrato.")
else:
    print("Dati di training per baseline mancanti o vuoti.")