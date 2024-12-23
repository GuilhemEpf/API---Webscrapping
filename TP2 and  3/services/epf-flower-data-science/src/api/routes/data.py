import os
import kagglehub
import pandas as pd
from fastapi import APIRouter, HTTPException
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import opendatasets as od
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from src.firestore import FirestoreClient
from fastapi import APIRouter, HTTPException
from src.firestore import FirestoreClient
from pydantic import BaseModel

# Initialize the FastAPI router
router = APIRouter()
firestore_client = FirestoreClient()

# File paths
FILE_PATH = 'iris/iris/Iris.csv'
processed_path = "iris/iris/iris_processed.csv"
TRAIN_TEST_PATH = "iris/iris/train_test.json"
MODEL_PARAMS_PATH = "src/config/model_parameters.json"
MODEL_SAVE_PATH = "iris/models/iris_model.pkl"

# Download the Iris dataset from Kaggle
def download_iris_data():
    od.download("https://www.kaggle.com/uciml/iris", force=True, data_dir='iris')
    path = 'iris/iris/Iris.csv'
    print(f"Dataset downloaded to: {path}")
    return path

@router.get("/download")
def download_data():
    try:
        path = download_iris_data()  # Call the function to download the dataset
        return {"message": "Dataset downloaded successfully!", "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading the dataset: {str(e)}")

# Load and return the Iris dataset as JSON
@router.get("/load")
def load_iris_dataset():
    if not os.path.exists(FILE_PATH):
        print(FILE_PATH)
        raise HTTPException(status_code=404, detail="Iris dataset file not found.")
    
    # Load the dataset into a DataFrame
    try:
        df = pd.read_csv(FILE_PATH)
        # Convert the DataFrame to a dictionary (JSON format)
        data = df.to_dict(orient='records')
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading the dataset: {str(e)}")

@router.get("/process")
def process_iris_data():
    """
    Process the Iris dataset to prepare it for model training.
    """
    if not os.path.exists(FILE_PATH):
        raise HTTPException(status_code=404, detail="Iris dataset file not found.")
    
    try:
        df = pd.read_csv(FILE_PATH)
        # Rename columns to lowercase and replace spaces with underscores
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        # Add an encoded column for 'species'
        df["species_encoded"] = df["species"].astype("category").cat.codes
        df.to_csv(processed_path, index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the dataset: {str(e)}")
    
    return {"message": "Dataset processed and saved as iris_processed.csv"}

@router.get("/split")
def split_iris_data():
    if not os.path.exists(processed_path):
        raise HTTPException(status_code=404, detail="The processed Iris dataset file is not found.")
    try:
        df = pd.read_csv(processed_path)
        X = df.drop(columns=['species', 'species_encoded'])
        y = df['species_encoded']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        train_test_data = {
            "train": {"features": X_train.values.tolist(), "target": y_train.tolist()},
            "test": {"features": X_test.values.tolist(), "target": y_test.tolist()}
        }
        os.makedirs(os.path.dirname(TRAIN_TEST_PATH), exist_ok=True)
        with open(TRAIN_TEST_PATH, "w") as file:
            json.dump(train_test_data, file)
        return {"message": "Data split into train and test sets and saved.", "path": TRAIN_TEST_PATH}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error splitting the dataset: {str(e)}")

@router.get("/train")
def train_iris_model():
    """
    Train a classification model on the Iris dataset and save the trained model.
    """
    # Check if required files exist
    if not os.path.exists(TRAIN_TEST_PATH):
        raise HTTPException(status_code=404, detail="Training/test data file not found.")
    
    if not os.path.exists(MODEL_PARAMS_PATH):
        raise HTTPException(status_code=404, detail="Model parameters file not found.")
    
    try:
        # Load training and test data
        with open(TRAIN_TEST_PATH, "r") as file:
            data = json.load(file)
        
        X_train = data["train"]["features"]
        y_train = data["train"]["target"]
        
        # Load model parameters
        with open(MODEL_PARAMS_PATH, "r") as file:
            model_params = json.load(file)
        
        # Create and train the model
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # Save the trained model
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        joblib.dump(model, MODEL_SAVE_PATH)
        
        return {"message": "Model trained and saved successfully.", "model_path": MODEL_SAVE_PATH}
    except NotFittedError as e:
        raise HTTPException(status_code=500, detail=f"Model training error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/predict")
def predict_iris():
    """
    Predict the species of Iris flowers automatically using pre-existing data.
    No need to provide features in the request; the API will predict using a sample.
    """
    try:
        # charge trained model
        model = joblib.load(MODEL_SAVE_PATH)
        
        # Charge processed data
        df = pd.read_csv(processed_path)
        
        # select a  sample to predict on (first sample) 
        sample = df.drop(columns=['species', 'species_encoded']).iloc[0].values.reshape(1, -1)  # Par exemple, premier échantillon
        
        # predict with the model
        prediction = model.predict(sample)
        
        # map the prediction to the species name
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        predicted_species = species_map.get(prediction[0], "Unknown")
        
        # return the prediction and sample features
        return {
            "sample_features": sample.tolist(),  # return the sample features
            "predicted_species": predicted_species  # return the predicted species
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

class UpdateParameters(BaseModel):
    n_estimators: int = None
    criterion: str = None
# Endpoint to create initial parameters
@router.post("/firestore/create")
def create_parameters():
    """Create the Firestore document with default parameters."""
    default_params = {"n_estimators": 100, "criterion": "gini"}
    return firestore_client.save_parameters(default_params)

# Endpoint to retrieve parameters
@router.get("/firestore/retrieve")
def retrieve_parameters():
    """Retrieve parameters from Firestore."""
    return firestore_client.get_parameters()

# Endpoint to update parameters
@router.put("/firestore/update")
def update_parameters(updates: UpdateParameters = None):
    """Update specific parameters in Firestore."""
    try:
        if updates is None:
            # Use default parameters from create_parameters if no updates are provided
            default_params = {"n_estimators": 100, "criterion": "gini"}
            return firestore_client.update_parameters(default_params)
        else:
            updates_dict = updates.dict(exclude_unset=True)
            return firestore_client.update_parameters(updates_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))