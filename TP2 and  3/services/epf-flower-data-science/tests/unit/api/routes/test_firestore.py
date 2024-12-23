import pytest
from fastapi.testclient import TestClient
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

def test_update_parameters():
    updates = {"n_estimators": 200}
    response = client.put("/firestore/update", json=updates)
    assert response.status_code == 200
    assert response.json() == {"message": "Parameters updated successfully."}