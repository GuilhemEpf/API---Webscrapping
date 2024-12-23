import sys
import os
import pytest
from fastapi.testclient import TestClient

# Ajouter le répertoire racine au sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../../../services/epf-flower-data-science")

from main import app  # Assurez-vous que 'main' est le nom de votre fichier principal contenant l'application FastAPI

client = TestClient(app)

def test_create_parameters():
    response = client.post("/firestore/create")
    assert response.status_code == 200
    assert response.json() == {"message": "Parameters saved successfully."}

def test_retrieve_parameters():
    response = client.get("/firestore/retrieve")
    assert response.status_code == 200
    assert "n_estimators" in response.json()
    assert "criterion" in response.json()

def test_update_parameters_with_defaults():
    response = client.put("/firestore/update")
    assert response.status_code == 200
    assert response.json() == {"message": "Parameters updated successfully."}

def test_update_parameters_with_body():
    updates = {"n_estimators": 200}
    response = client.put("/firestore/update", json=updates)
    assert response.status_code == 200
    assert response.json() == {"message": "Parameters updated successfully."}