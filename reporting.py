import pickle
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import scoring
import ingestion


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 
output_model_path = config['output_model_path']
dataset_csv_path = config['output_folder_path']
test_data_path = config['test_data_path']
model_path = config['test_data_path']
prod_deployment_path = config['prod_deployment_path'] 

##############Function for reporting
def score_model():
    df_new = ingestion.merge_multiple_dataframe(test_data_path)
    with open(os.path.join(prod_deployment_path,"trainedmodel.pkl"),"rb") as f:
        model = pickle.load(f)
    ### ACTUALLY WHAT ELSE OTHER REPORTS CAN ADD ???
    X_test = df_new.drop(["name","value_eur","wage_eur"], axis=1).values.reshape(-1,70)
    y_test = df_new["wage_eur"].values.reshape(-1,1).ravel()    
    pred = model.predict(X_test)
    score = model.score(X_test,y_test)   
    print(score) 
    #cm = metrics.confusion_matrix(pred,y_test)
    #sns.heatmap(cm, annot=True)
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    #plt.savefig(os.path.join(output_model_path,f"confusionmatrix_{date}.png"))
    
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace


if __name__ == '__main__':
    score_model()
