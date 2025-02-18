import pytest
import subprocess
import time

def test_dashboard_runs():
    """Vérifie que le dashboard Streamlit démarre sans erreur"""
    process = subprocess.Popen(["streamlit", "run", "src/Dashboard.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Attendre un peu pour laisser le dashboard démarrer
    time.sleep(10)

    # Vérifier si le processus est toujours en cours (c'est bon signe)
    assert process.poll() is None

    # Arrêter le processus après le test
    process.terminate()
