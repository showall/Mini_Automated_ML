
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import ingestion
##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = config['output_folder_path'] 
test_data_path = config['test_data_path'] 
prod_deployment_path = config['prod_deployment_path'] 
model_path = config['output_model_path'] 

##################Function to get model predictions
def model_predictions(dataset_file_location):
    with open(os.path.join(prod_deployment_path,"trainedmodel.pkl"),"rb") as f:
        model = pickle.load(f)

    df_new = ingestion.merge_multiple_dataframe(test_data_path).drop(["name","value_eur","wage_eur"], axis=1)
    result = []
    for index,row in df_new.iterrows():
        result.append(model.predict(row.values.reshape(-1,70)))    
    #read the deployed model and a test dataset, calculate predictions
    return str(result) #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    df = pd.read_csv(os.path.join(dataset_csv_path,"finaldata.csv"))
    summary={}
    for col in df.columns :
        if df[col].dtypes != object and df[col].dtypes != bool and df[col].dtypes != []:
            x = pd.to_numeric(df[col])
            summary[col]=[np.mean(x),np.median(x),np.std(x)]
    #calculate summary statistics here
    #print (summary)    
    return summary
    #return value should be a list containing all summary statistics

def missing_data_check():
    df = pd.read_csv(os.path.join(dataset_csv_path,"finaldata.csv"))
    count = [df[col].isna().sum()/len(df[col]) for col in df.columns]
    return {"summary":count}

##################Function to get timings
def execution_time():
    list_1 = []
    for i in ["training.py","ingestion.py"]:
        starttime = timeit.default_timer()
        os.system(f"python {i}")
        endtime = timeit.default_timer()
        list_1.append(endtime-starttime)
    #calculate timing of training.py and ingestion.py
    return list_1#list_1#return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    dep = subprocess.check_output(["pip","list","--outdated"])
    return dep

if __name__ == '__main__':
    model_predictions("testdata.csv")
    dataframe_summary()
    missing_data_check()
    execution_time()
    outdated_packages_list()





    
