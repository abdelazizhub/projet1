#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
nohup uvicorn api:app --host 0.0.0.0 --port 8000 &
python api.py
streamlit run Dashboard.py