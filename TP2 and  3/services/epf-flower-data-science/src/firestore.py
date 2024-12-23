import os
import google.auth
from google.cloud import firestore
from google.oauth2 import service_account


class FirestoreClient:
    """Wrapper around a database"""

    client: firestore.Client

    def __init__(self) -> None:
        """Init the client."""
        credentials_path = os.path.join(os.path.dirname(__file__), 'config', 'Firebase_credentials.json')        
        credentials = service_account.Credentials.from_service_account_file(credentials_path)

        self.client = firestore.Client(credentials=credentials)

    def get(self, collection_name: str, document_id: str) -> dict:
        """Find one document by ID.
        Args:
            collection_name: The collection name
            document_id: The document id
        Return:
            Document value.
        """
        doc = self.client.collection(collection_name).document(document_id).get()
        if doc.exists:
            return doc.to_dict()
        raise FileExistsError(
            f"No document found at {collection_name} with the id {document_id}"
        )

    def save_parameters(self, params: dict) -> dict:
        """Save parameters to Firestore."""
        doc_ref = self.client.collection("parameters").document("parameters")
        doc_ref.set(params)
        return {"message": "Parameters saved successfully."}

    def get_parameters(self) -> dict:
        """Retrieve parameters from Firestore."""
        return self.get("parameters", "parameters")

    def update_parameters(self, updates: dict) -> dict:
        """Update specific parameters in Firestore."""
        doc_ref = self.client.collection("parameters").document("parameters")
        doc_ref.update(updates)
        return {"message": "Parameters updated successfully."}