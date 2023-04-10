import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
import json
import ingestion



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
model_path = os.path.join(config['output_model_path']) 

#################Function for model scoring
def score_model(test_data_path):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    with open(os.path.join(model_path,"trainedmodel.pkl"),"rb") as f:
        model = pickle.load(f)

    df_new = ingestion.merge_multiple_dataframe(test_data_path)
    X_test = df_new.drop(["name","value_eur","wage_eur"], axis=1).values.reshape(-1,70)
    y_test = df_new["wage_eur"].values.reshape(-1,1).ravel()
    pred = model.predict(X_test)
    score = model.score(X_test,y_test)
    with open(os.path.join(config['output_model_path'],"latestscore.txt"),"w") as f:
        f.write(str(score))
    return score    

if __name__ == "__main__":
    score_model(test_data_path)