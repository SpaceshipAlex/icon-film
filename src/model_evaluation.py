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
import joblib

print("Caricamento dei dati e dei modelli per la valutazione...")

if os.path.exists("models/testTargetsReg.npy"):
    testTargetsReg = np.load("models/testTargetsReg.npy")
    print("testTargetsReg caricato")
if os.path.exists("models/testTargetsClass.npy"):
    testTargetsClass = np.load("models/testTargetsClass.npy")
    print("testTargetsClass caricato")   
if os.path.exists("models/testBaselineFeatures.joblib"):
    testBaselineFeatures = joblib.load("models/testBaselineFeatures.joblib")
    print("testBaselineFeatures caricato")
if os.path.exists("kernel/KTest_filtered.npy"):
    KTest = np.load("kernel/KTest_filtered.npy")
    print("KTest filtrata caricata")
testIris = []
if os.path.exists("kernel/testFilmIris_filtered.txt"):
    with open("kernel/testFilmIris_filtered.txt", "r") as f:
        testIris = [line.strip() for line in f if line.strip()]
    print("testIris filtrata caricata")

modelNames = {
    'rfRegressor': 'models/rfRegressorBaseline.joblib',
    'rfClassifier': 'models/rfClassifierBaseline.joblib',
    'svr': 'models/svr.joblib',
    'svc': 'models/svc.joblib'
}
models = {}
for modelKey, modelFile in modelNames.items():
    if os.path.exists(modelFile):
        try:
            models[modelKey] = joblib.load(modelFile)
            print(f"Modello {modelKey} caricato")
        except Exception as e:
            print(f"Errore durante il caricamento di {modelKey}")
            models[modelKey] = None
    else:
        models[modelKey] = None

print("Verifica di allineamento dati di test...")