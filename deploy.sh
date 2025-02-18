#!/bin/bash
scp -i "projet.pem" api.py ubuntu@15.188.51.13:/home/ubuntu/
scp -i "projet.pem" Dashboard.py ubuntu@15.188.51.13:/home/ubuntu/
scp -i "projet.pem" requirements.txt ubuntu@15.188.51.13:/home/ubuntu/
scp -i "projet.pem" runapi.sh ubuntu@15.188.51.13:/home/ubuntu/
scp -i "projet.pem" -r C:/Users/Aziz/Documents/projet_7/mlflow_model ubuntu@15.188.51.13:/home/ubuntu/
scp -i "projet.pem" -r C:/Users/Aziz/Documents/projet_7/mlruns ubuntu@15.188.51.13:/home/ubuntu/
scp -i "projet.pem" test_df_sample.csv ubuntu@15.188.51.13:/home/ubuntu/
scp -i "projet.pem" train_df_sample.csv ubuntu@15.188.51.13:/home/ubuntu/
ssh -i "projet.pem" ubuntu@15.188.51.13