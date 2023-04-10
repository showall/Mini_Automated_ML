import pandas as pd
import numpy as np
import pickle
import os
import json
from pathlib import Path


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
p = Path(prod_deployment_path)
p.mkdir(exist_ok=True)

model_path = os.path.join(config['output_model_path']) 

####################function for deployment
def store_model_into_pickle(model_name):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    
    with open(os.path.join(model_path,model_name),"rb") as f:
        model = pickle.load(f)
    #COPY TO DEPLOYMENT FOLDER
    with open(os.path.join(p,model_name),"wb") as f:
        pickle.dump(model,f)

    with open(os.path.join(model_path,"latestscore.txt"),"r") as f:
        latestscore = f.readlines()              
        
    with open(os.path.join(dataset_csv_path,"ingestedfiles.txt"),"r") as f:
        ingestfiles = f.readlines()                
    #COPY TO DEPLOYMENT FOLDER
    with open(os.path.join(p,"latestscore.txt"),"w") as f:
        for row in latestscore:
            f.write(row)     
    #COPY TO DEPLOYMENT FOLDER
    with open(os.path.join(p,"ingestfiles.txt"),"w") as f:
        for row in ingestfiles:
            f.write(row)  

if __name__ == "__main__":
    store_model_into_pickle("trainedmodel.pkl")