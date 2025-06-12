import numpy as np
import os
from owlready2 import *
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
import pandas as pd 
import joblib

import matplotlib.pyplot as plt

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
def getTargetsFromKB(iris, qualityThreshold = 7):
    targetsRegression = [] # Memorizza le valutazioni dei film per la regressione
    targetsClassification = [] # Memorizza le etichette binarie per la valutazione di un film (alta/bassa qualità) per la classificazione
    validIris = [] # IRI dei film per cui si è trovato un target valido

    for iri in iris:
        film = onto.search_one(iri = f"*{iri.split('#')[1]}")
        if film and film.tmdbRating:
            rating = film.tmdbRating
            targetsRegression.append(rating)
            targetsClassification.append(1 if rating >= qualityThreshold else 0)
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

# Effettua GridSearchCV per trovare il modello SVR con i migliori iperparametri
def optimizeSVR(KTrain, trainTargetsReg, cv = 5, n_jobs = -1):
    params = { # iperparametri tra cui effettuare la scelta del migliore, provando tutte le combinazioni
        'C': [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
        'epsilon': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
        'gamma': ['scale']
    }
    svr = SVR(kernel = 'precomputed') # il kernel è stato già calcolato nel passo precedente
    maeScorer = make_scorer(mean_absolute_error, greater_is_better = False)

    gridSearch = GridSearchCV(
        estimator = svr,
        param_grid = params,
        scoring = maeScorer,
        cv = cv,
        n_jobs = n_jobs,
        verbose = 1,
        return_train_score = True
    )

    print("Ottimizzazione SVR in corso...")
    gridSearch.fit(KTrain, trainTargetsReg)
    print(f"Migliori parametri SVR: {gridSearch.best_params_}")
    print(f"Miglior MAE: {-gridSearch.best_score_:.4f}")

    return gridSearch.best_estimator_

# Effettua GridSearchCV per trovare il modello SVC con i migliori iperparametri
def optimizeSVC(KTrain, trainTargetsClass, cv = 5, n_jobs = -1):
    params = {
        'C': [0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
        'gamma': ['scale'],
        'class_weight': [None, 'balanced'],
        'probability': [True]
    }
    svc = SVC(kernel = 'precomputed')
    stratifiedCv = StratifiedKFold(n_splits = cv, shuffle = True, random_state = 3)

    gridSearch = GridSearchCV(
        estimator = svc,
        param_grid = params,
        scoring = 'f1',
        cv = stratifiedCv,
        n_jobs = n_jobs,
        verbose = 1,
        return_train_score = True
    )

    print("Ottimizzazione SVC in corso...")
    gridSearch.fit(KTrain, trainTargetsClass)
    print(f"Migliori parametri SVC: {gridSearch.best_params_}")
    print(f"Miglior F1: {-gridSearch.best_score_:.4f}")

    return gridSearch.best_estimator_

# Effettua GridSearchCV per trovare il modello Baseline Regressor (RandomForest) con i migliori iperparametri
def optimizeRFReg(trainBaselineFeatures, trainTargetsReg, preprocessor, cv = 5, n_jobs = -1):
    params = {
        'regressor__n_estimators': [50, 100, 300, 500],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 10],
        'regressor__min_samples_leaf': [1, 4],
        'regressor__max_features': ['sqrt', 0.5],
        'regressor__bootstrap': [True, False],
        'regressor__max_samples': [None, 0.8], 
        'regressor__ccp_alpha': [0.0, 0.01]
    }
    rfRegressor = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state = 3))
    ])
    maeScorer = make_scorer(mean_absolute_error, greater_is_better = False)


    gridSearch = GridSearchCV(
        estimator = rfRegressor,
        param_grid = params,
        scoring = maeScorer,
        cv = cv,
        n_jobs = n_jobs,
        verbose = 1,
        return_train_score = True
    )

    print("Ottimizzazione RF Regressor in corso...")
    gridSearch.fit(trainBaselineFeatures, trainTargetsReg)
    print(f"Migliori parametri RF Regressor: {gridSearch.best_params_}")
    print(f"Miglior MAE: {-gridSearch.best_score_:.4f}")

    return gridSearch.best_estimator_


# Effettua GridSearchCV per trovare il modello Baseline Classifier (RandomForest) con i migliori iperparametri
def optimizeRFClass(trainBaselineFeatures, trainTargetsClass, preprocessor, cv = 5, n_jobs = -1):
    params = {
        'regressor__n_estimators': [50, 100, 300, 500],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 10],
        'regressor__min_samples_leaf': [1, 4],
        'regressor__max_features': ['sqrt', 0.5],
        'regressor__bootstrap': [True, False],
        'regressor__max_samples': [None, 0.8], 
        'regressor__ccp_alpha': [0.0, 0.01],
        'regressor__class_weight': [None, 'balanced']
    }
    rfRegressor = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('regressor', RandomForestClassifier(random_state = 3))
    ])
    stratifiedCv = StratifiedKFold(n_splits = cv, shuffle = True, random_state = 3)


    gridSearch = GridSearchCV(
        estimator = rfRegressor,
        param_grid = params,
        scoring = 'f1',
        cv = stratifiedCv,
        n_jobs = n_jobs,
        verbose = 1,
        return_train_score = True
    )

    print("Ottimizzazione RF Classifier in corso...")
    gridSearch.fit(trainBaselineFeatures, trainTargetsClass)
    print(f"Migliori parametri RF Classifier: {gridSearch.best_params_}")
    print(f"Miglior F1: {-gridSearch.best_score_:.4f}")

    return gridSearch.best_estimator_
    
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
testTargetsReg, testTargetsClass, testValidIris = getTargetsFromKB(testIris)
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
print(f"Target di test: {len(testTargetsReg)} (regressione), {len(testTargetsClass)} (classificazione)")

if KTrain is not None and (KTrain.shape[0] != len(trainTargetsReg) or (KTrain.shape[1] != len(trainTargetsReg))):
    print("La dimensione di KTrain è diversa dal numero di target di training dopo il filtraggio, errore.")
    exit()
elif KTest is not None and KTrain is not None and (KTest.shape[0] != len(testTargetsReg) or (KTest.shape[1] != KTrain.shape[0])):
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

# Ottengo i 4 modelli usando GridSearchCV per ognuno
rfRegressor = optimizeRFReg(trainBaselineFeatures, trainTargetsReg, baselinePreprocessor)
rfClassifier = optimizeRFClass(trainBaselineFeatures, trainTargetsClass, baselinePreprocessor)
svr = optimizeSVR(KTrain, trainTargetsReg)
svc = optimizeSVC(KTrain, trainTargetsClass)

# Salvataggio dei modelli, se sono stati addestrati (hanno attributo steps per i baseline e support_vectors_ per i modelli semantici)
if 'rfRegressor' in locals() and hasattr(rfRegressor, 'steps'):
    joblib.dump(rfRegressor, 'models/rfRegressorBaseline.joblib')
if 'rfClassifier' in locals() and hasattr(rfClassifier, 'steps'):
    joblib.dump(rfClassifier, 'models/rfClassifierBaseline.joblib') 
if 'svr' in locals() and hasattr(svr, 'support_vectors_'):
    joblib.dump(svr, 'models/svr.joblib')
if 'svc' in locals() and hasattr(svc, 'support_vectors_'):
    joblib.dump(svc, 'models/svc.joblib')

# Salvataggio dei dati per la valutazione

if 'testTargetsReg' in locals() and testTargetsReg.size > 0:
    np.save("models/testTargetsReg.npy", testTargetsReg)
if 'testTargetsClass' in locals() and testTargetsClass.size > 0:
    np.save("models/testTargetsClass.npy", testTargetsClass)
if 'testBaselineFeatures' in locals() and not testBaselineFeatures.empty:
    joblib.dump(testBaselineFeatures, "models/testBaselineFeatures.joblib")
if 'KTest' in locals() and KTest is not None and KTest.size > 0:
    np.save("kernel/KTest_filtered.npy", KTest)
if 'testIris' in locals() and testIris:
    with open("kernel/testFilmIris_filtered.txt", "w") as f:
        for iri in testIris:
            f.write(f"{iri}\n")