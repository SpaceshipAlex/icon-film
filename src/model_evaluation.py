import numpy as np
import os
from owlready2 import *
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
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
for modelKey, modelFile in modelNames.items(): # Carico i modelli salvati nel passo precedente
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
if testTargetsReg.size > 0 and not testBaselineFeatures.empty and len(testTargetsReg) != testBaselineFeatures.shape[0]:
    print(f"Attenzione! Mismatch tra testTargetsReg ({len(testTargetsReg)} e testBaselineFeatures {testBaselineFeatures.shape[0]})")
if testTargetsClass.size > 0 and not testBaselineFeatures.empty and len(testTargetsClass) != testBaselineFeatures.shape[0]:
    print(f"Attenzione! Mismatch tra testTargetsClass ({len(testTargetsClass)} e testBaselineFeatures {testBaselineFeatures.shape[0]})")
if testTargetsReg.size > 0 and KTest is not None and len(testTargetsReg) != KTest.shape[0]:
    print(f"Attenzione! Mismatch tra testTargetsReg ({len(testTargetsReg)}) e KTest ({KTest.shape[0]})")
if testTargetsClass.size > 0 and KTest is not None and len(testTargetsClass) != KTest.shape[0]:
    print(f"Attenzione! Mismatch tra testTargetsClass ({len(testTargetsClass)}) e KTest ({KTest.shape[0]})")
print("Verifica completata.")

results = {}

if 'testTargetsReg' in locals() and testTargetsReg.size > 0:
    print("Valutazione regressione...")

    # Valuto il modello baseline (RandomForest)
    if models.get('rfRegressor') and 'testBaselineFeatures' in locals() and not testBaselineFeatures.empty:
        try:
            pred = models.get('rfRegressor').predict(testBaselineFeatures)
            maeBaseline = mean_absolute_error(testTargetsReg, pred)
            rmseBaseline = root_mean_squared_error(testTargetsReg, pred)
            r2Baseline = r2_score(testTargetsReg, pred)
            results['baselineReg'] = {'MAE': maeBaseline, 'RMSE': rmseBaseline, 'R2': r2Baseline}
            print(f"Baseline RF Regressor: MAE {maeBaseline:.4f}, RMSE {rmseBaseline:.4f}, R2 {r2Baseline:.4f}")
        except Exception as e:
            print(f"Errore durante la valutazione del Baseline RF Regressor: {e}")
            results['baselineReg'] = {'MAE': float('nan'), 'RMSE': float('nan'), 'R2': float('nan')}
    else:
        print("Baseline RF Regressor non addestrato o dati di test mancanti")
    
    # Valuto il modello semantico (SVR)
    if models.get('svr') and 'KTest' in locals() and KTest is not None and KTest.size > 0:
        if KTest.shape[0] == len(testTargetsReg):
            try:
                pred = models.get('svr').predict(KTest)
                maeSemantic = mean_absolute_error(testTargetsReg, pred)
                rmseSemantic = root_mean_squared_error(testTargetsReg, pred)
                r2Semantic = r2_score(testTargetsReg, pred)
                results['svrReg'] = {'MAE': maeSemantic, 'RMSE': rmseSemantic, 'R2': r2Semantic}
                print(f"Semantic SVR: MAE {maeSemantic:.4f}, RMSE {rmseSemantic:.4f}, R2 {r2Semantic:.4f}")
            except Exception as e:
                print(f"Errore durante la valutazione del Semantic SVR: {e}")
                results['svrReg'] = {'MAE': float('nan'), 'RMSE': float('nan'), 'R2': float('nan')}
        else:
            print(f"Attenzione! Mismatch tra KTest ({KTest.shape[0]}) e testTargetsReg ({len(testTargetsReg)}) per SVR.")
    else:
        print("Semantic SVR non addestrato o KTest mancante.")
else:
    print("Target di test per la regressione non disponibili.")

if 'testTargetsClass' in locals() and testTargetsClass.size > 0:
    print("Valutazione classificazione...")

    # Valuto il modello Baseline (RandomForest)
    if models.get('rfClassifier') and 'testBaselineFeatures' in locals() and not testBaselineFeatures.empty:
        try:
            pred = models.get('rfClassifier').predict(testBaselineFeatures)
            predProba = models.get('rfClassifier').predict_proba(testBaselineFeatures)[:, 1]

            accBaseline = accuracy_score(testTargetsClass, pred)
            f1Baseline = f1_score(testTargetsClass, pred, average = 'binary')
            precisionBaseline = precision_score(testTargetsClass, pred, average = 'binary')
            recallBaseline = recall_score(testTargetsClass, pred, average = 'binary')
            aucBaseline = roc_auc_score(testTargetsClass, predProba) if len(np.unique(testTargetsClass)) > 1 else float('nan')
            results['baselineClass'] = {'Accuracy': accBaseline, 'F1': f1Baseline, 'Precision': precisionBaseline, 'Recall': recallBaseline, 'AUC': aucBaseline}
            print(f"Baseline RF Classifier: Accuracy {accBaseline:.4f}, F1 {f1Baseline:.4f}, Precision {precisionBaseline:.4f}, Recall {recallBaseline:.4f}, AUC {aucBaseline:.4f}")
        except Exception as e:
            print(f"Errore durante la valutazione del Baseline RF Classifier: {e}")
            results['baselineClass'] = {'Accuracy': float('nan'), 'F1': float('nan'), 'Precision': float('nan'), 'Recall': float('nan'), 'AUC': float('nan')}
    else:
        print("Baseline RF Classifier non adddestrato o dati di test mancanti.")

    # Valuto il modello semantico (SVC)
    if models.get('svc') and 'KTest' in locals() and KTest is not None and KTest.size > 0:
        if KTest.shape[0] == len(testTargetsClass):
            try:
                pred = models.get('svc').predict(KTest)
                predProba = models.get('svc').predict_proba(KTest)[:, 1]

                accSemantic = accuracy_score(testTargetsClass, pred)
                f1Semantic = f1_score(testTargetsClass, pred, average = 'binary')
                precisionSemantic = precision_score(testTargetsClass, pred, average = 'binary')
                recallSemantic = recall_score(testTargetsClass, pred, average = 'binary')
                aucSemantic = roc_auc_score(testTargetsClass, predProba) if len(np.unique(testTargetsClass)) > 1 else float('nan')
                results['svc'] = {'Accuracy': accSemantic, 'F1': f1Semantic, 'Precision': precisionSemantic, 'Recall': recallSemantic, 'AUC': aucSemantic}
                print(f"Semantic SVC: Accuracy {accSemantic:.4f}, F1 {f1Semantic:.4f}, Precision {precisionSemantic:.4f}, Recall {recallSemantic:.4f}, AUC {aucSemantic:.4f}")
            except Exception as e:
                print(f"Errore durante la valutazione del Semantic SVC: {e}")
                results['svc'] = {'Accuracy': float('nan'), 'F1': float('nan'), 'Precision': float('nan'), 'Recall': float('nan'), 'AUC': float('nan')}
        else:
            print(f"Attenzione! Mismatch tra KTest ({KTest.shape[0]}) e testTargetsReg ({len(testTargetsReg)}) per SVC.")
    else:
        print("Semantic SVC non addestrato o KTest mancante.")
else:
    print("Target di test per la classificazione non disponibili.")

# Stampo un riassunto
print("Riassunto performance:")
if results:
    resultsDf = pd.DataFrame.from_dict(results, orient = 'index')
    print(resultsDf.to_string())
else:
    print("Nessun risultato da mostrare.")

