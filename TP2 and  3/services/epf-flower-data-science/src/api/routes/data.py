import os
import kagglehub
import pandas as pd
from fastapi import APIRouter, HTTPException
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import opendatasets as od 
# Initialiser le routeur FastAPI
router = APIRouter()

# Dossier où le dataset Iris a été téléchargé
DATA_DIR = 'API---Webscrapping\TP2 and  3\services\epf-flower-data-science\src\data'
FILE_PATH = os.path.join(DATA_DIR, 'iris.csv')  # Chemin du fichier CSV du dataset


# Télécharger le dataset Iris depuis Kaggle
def download_iris_data():
    path = od.download("https://www.kaggle.com/datasets/uciml/iris", path=DATA_DIR)
    print(f"Dataset téléchargé à l'emplacement : {path}")
    return path


@router.get("/download")
def download_data():

    try:
        path = download_iris_data()  # Appel de la fonction pour télécharger le dataset
        return {"message": "Dataset téléchargé avec succès!", "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du téléchargement du dataset: {str(e)}")


# Charger et retourner le dataset Iris sous forme de JSON
@router.get("/load")
def load_iris_dataset():

    if not os.path.exists(FILE_PATH):
        raise HTTPException(status_code=404, detail="Le fichier du dataset Iris est introuvable.")
    
    # Charger le dataset dans un DataFrame
    try:
        df = pd.read_csv(FILE_PATH)
        # Convertir le DataFrame en dictionnaire (format JSON)
        data = df.to_dict(orient='records')
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du dataset: {str(e)}")


@router.get("/process")
def process_iris_data():
    """
    Effectue le traitement nécessaire sur le dataset Iris avant l'entraînement d'un modèle.
    Ici, nous appliquons une normalisation des données.
    """
    if not os.path.exists(FILE_PATH):
        raise HTTPException(status_code=404, detail="Le fichier du dataset Iris est introuvable.")
    
    try:
        # Charger le dataset dans un DataFrame
        df = pd.read_csv(FILE_PATH)
        
        # Séparer les features et la cible
        X = df.drop('species', axis=1)  # Features
        y = df['species']  # Cible
        
        # Appliquer une normalisation (Standardisation)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Retourner les données normalisées
        return {"features": X_scaled.tolist(), "target": y.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du dataset: {str(e)}")

@router.get("/split")
def split_iris_data():
    """
    Divise le dataset Iris en ensembles d'entraînement et de test, puis retourne les ensembles sous forme de JSON.
    """
    if not os.path.exists(FILE_PATH):
        raise HTTPException(status_code=404, detail="Le fichier du dataset Iris est introuvable.")
    
    try:
        # Charger le dataset
        df = pd.read_csv(FILE_PATH)
        
        # Séparer les features et la cible
        X = df.drop('species', axis=1)
        y = df['species']
        
        # Diviser le dataset en train et test (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convertir les DataFrames en listes pour le JSON
        train_data = {"features": X_train.values.tolist(), "target": y_train.tolist()}
        test_data = {"features": X_test.values.tolist(), "target": y_test.tolist()}
        
        return {"train": train_data, "test": test_data}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la division des données: {str(e)}")
