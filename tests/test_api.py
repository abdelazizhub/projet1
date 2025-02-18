from fastapi.testclient import TestClient
from src.api import app  # Assurez-vous que cet import fonctionne

from fastapi.testclient import TestClient
from starlette.applications import Starlette

# Assurez-vous que app est bien une instance de FastAPI
app = app if isinstance(app, Starlette) else app

client = TestClient(app)

def test_read_root():
    """Vérifie que l'endpoint racine retourne le bon message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Bienvenu (e) dans notre API"

def test_check_client_id():
    """Teste si un client existe dans la base de données."""
    client_id = 192535  # Remplace par un ID existant dans ton `test_df_sample.csv`
    response = client.get(f"/{client_id}")
    assert response.status_code == 200
    assert isinstance(response.json(), bool)

def test_get_prediction():
    """Teste si l'endpoint /prediction/{client_id} retourne une valeur valide."""
    client_id = 192535  # Remplace par un ID valide
    response = client.get(f"/prediction/{client_id}")
    assert response.status_code == 200
    assert isinstance(response.json(), float)  # Une probabilité entre 0 et 1

import json  # Ajout du module JSON

import json

def test_get_clients_similaires():
    """Teste si l'endpoint /clients_similaires/{client_id} retourne un JSON valide."""
    client_id = 192535  # Remplace par un ID existant
    response = client.get(f"/clients_similaires/{client_id}")
    
    assert response.status_code == 200

    # Vérification du type de réponse
    if isinstance(response.json(), str):  
        json_data = json.loads(response.json())  # Convertir en dict si string
    else:
        json_data = response.json()  # Utiliser directement si déjà un dict

    assert isinstance(json_data, dict)  # Vérifier que c'est bien un dictionnaire
    assert "SK_ID_CURR" in json_data  # Vérifier que la clé principale est bien présente
    
def test_shap_values_local():
    """Teste si les valeurs SHAP locales sont bien retournées."""
    client_id = 192535  # Remplace par un ID valide
    response = client.get(f"/shaplocal/{client_id}")
    assert response.status_code == 200
    data = response.json()
    assert "shap_values" in data
    assert "base_value" in data
    assert "data" in data
    assert "feature_names" in data

def test_shap_global():
    """Teste si les valeurs SHAP globales sont bien retournées."""
    response = client.get("/shap/")
    assert response.status_code == 200
    data = response.json()
    assert "shap_values_0" in data
    assert "shap_values_1" in data