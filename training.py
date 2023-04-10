import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import json
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = config['output_folder_path']
model_path =config['output_model_path']

#################Function for training the model
def train_model():

    df = pd.read_csv(os.path.join(dataset_csv_path,"finaldata.csv"))

    X = df.drop(["name","value_eur","wage_eur"], axis=1).values.reshape(-1,70)
    Y = df["wage_eur"].values.reshape(-1,1).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3)
    
    #use this logistic regression for training
    MODEL_pipeline = make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=10))

    #fit the logistic regression to your data
    model =  MODEL_pipeline.fit(X_train,y_train)
    #print()
   # print(model.predict(X_test[0:1]))
  #  print(y_test[0:1])
    #write the trained model to your workspace in a file called trainedmodel.pkl
    p = Path(model_path)
    p.mkdir(exist_ok=True)
    with open(os.path.join(p,"trainedmodel.pkl"),"wb") as f: 
        pickle.dump(model,f)

    with open(os.path.join(p,"latestscore.txt"),"w") as f:
        f.write(str(model.score(X_test,y_test)))     

if __name__ == "__main__":
    train_model()