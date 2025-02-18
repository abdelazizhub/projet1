# Import des bibliothèques nécessaires
from fastapi import FastAPI
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle
import uvicorn
import shap


# Création d'une instance FastAPI
app = FastAPI()

# Chargement du modèle et des données
model = pickle.load(open('mlflow_model/model.pkl', 'rb'))
data = pd.read_csv('test_df_sample.csv')
data_train = pd.read_csv('train_df_sample.csv')

# Sélection des colonnes numériques pour la mise à l'échelle
cols = data.select_dtypes(['float64']).columns

# Mise à l'échelle des données de test
data_scaled = data.copy()
data_scaled[cols] = StandardScaler().fit_transform(data[cols])

# Mise à l'échelle des données d'entraînement
cols = data_train.select_dtypes(['float64']).columns
data_train_scaled = data_train.copy()
data_train_scaled[cols] = StandardScaler().fit_transform(data_train[cols])

# Initialisation de l'explainer Shapley pour les valeurs locales
explainer = shap.TreeExplainer(model['classifier'])

# Définition des points d'extrémité de l'API

@app.get('/')
def welcome():
    """Message de bienvenue."""
    return 'Bienvenu (e) dans notre API'

@app.get('/{client_id}')
def check_client_id(client_id: int):
    """Vérification de l'existence d'un client dans la base de données."""
    if client_id in list(data['SK_ID_CURR']):
        return True
    else:
        return False

@app.get('/prediction/{client_id}')
def get_prediction(client_id: int):
    """Calcul de la probabilité de défaut pour un client."""
    client_data = data[data['SK_ID_CURR'] == client_id]
    info_client = client_data.drop('SK_ID_CURR', axis=1)
    prediction = model.predict_proba(info_client)[0][1]
    return prediction

@app.get('/clients_similaires/{client_id}')
def get_data_voisins(client_id: int):
    """Calcul des clients similaires les plus proches."""
    features = list(data_train_scaled.columns)
    features.remove('SK_ID_CURR')
    features.remove('TARGET')

    # Entraînement du modèle NearestNeighbors
    nn = NearestNeighbors(n_neighbors=10, metric='euclidean')
    nn.fit(data_train_scaled[features])

    # Recherche des voisins du client
    reference_id = client_id
    reference_observation = data_scaled[data_scaled['SK_ID_CURR'] == reference_id][features].values
    indices = nn.kneighbors(reference_observation, return_distance=False)
    df_voisins = data_train.iloc[indices[0], :]

    return df_voisins.to_json()

from fastapi.responses import JSONResponse
import logging

logging.basicConfig(level=logging.INFO)

@app.get('/shaplocal/{client_id}')
async def shap_values_local(client_id: int):
    try:
        # Vérification client existant
        if client_id not in data['SK_ID_CURR'].values:
            return JSONResponse(
                status_code=404,
                content={"error": "Client non trouvé", "details": f"ID {client_id} inexistant"}
            )

        # Préparation des données
        client_row = data_scaled[data_scaled['SK_ID_CURR'] == client_id]
        if client_row.empty:
            return JSONResponse(
                status_code=404,
                content={"error": "Données client manquantes"}
            )
            
        client_data = client_row.drop('SK_ID_CURR', axis=1)
        logging.info(f"Shape des données client: {client_data.shape}")

        # Calcul SHAP
        explainer = shap.TreeExplainer(model['classifier'])
        shap_values = explainer.shap_values(client_data)
        
        # Sélection des valeurs SHAP pour la classe positive (ou unique)
        shap_values_class = (
            shap_values[1][0].tolist() if len(shap_values) > 1
            else shap_values[0][0].tolist()
        )
        # Pour la valeur de base, fais de même si nécessaire
        base_val = (
            float(explainer.expected_value[1]) if isinstance(explainer.expected_value, list) and len(explainer.expected_value) > 1
            else float(explainer.expected_value)
        )

        return {
            "shap_values": shap_values_class,
            "base_value": base_val,
            "data": client_data.values[0].tolist(),
            "feature_names": client_data.columns.tolist()
        }


    except Exception as e:
        logging.error(f"Erreur SHAP: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Erreur de calcul SHAP", "details": str(e)}
        )

@app.get('/shap/')
def shap_values():
    """Calcul des valeurs Shapley pour l'ensemble du jeu de données."""
    shap_val = explainer.shap_values(data_scaled.drop('SK_ID_CURR', axis=1))
    return {'shap_values_0': shap_val[0].tolist(),
            'shap_values_1': shap_val[1].tolist()}

# Démarrage du serveur FastAPI
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)