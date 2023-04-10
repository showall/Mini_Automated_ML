import os
import json
import ast
import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion
import pickle
import training
from sklearn import metrics
import pandas as pd

with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path']) 

##################Check and read new data
#first, read ingestedfiles.txt
with open(os.path.join(output_folder_path ,"ingestedfiles.txt"),"r") as f:
    ingestedfiles = ast.literal_eval(f.read())

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
currentfiles = [files for files in os.listdir(os.path.join(config['input_folder_path']))]
common = list(set(currentfiles)-set(ingestedfiles))
if common != []:
##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here    
    print("new files available")

    df = pd.DataFrame(columns= ["name","age","dob",
     "height_cm","weight_kg",
     "nationality","club","overall","potential",	
     "value_eur", "wage_eur","player_positions",
     "preferred_foot","international_reputation",
     "weak_foot","skill_moves", "work_rate", "versatility",
	"release_clause","talent", 
    "dominant_position", "traits"] )

    for file in common:
        df1 = pd.read_csv(os.path.join(input_folder_path,file), encoding='latin1')
        df = df.append(df1)
    df = df.drop_duplicates()
    df["potential_out_of_overall"] = round(df["overall"].astype("float")/df["potential"].astype("float"),2)
    df1 = pd.get_dummies(df["talent"])
    df2 = pd.get_dummies(df["traits"])
   # df_new = pd.concat([df, df1], axis=0)
   # df_new = pd.concat([df_new, df2], axis=0)
    df["bmi"] = df.weight_kg *10000 / (df.height_cm**2)
    df["bmi"] = round(df["bmi"].astype(float),2)
    df_club_type = pd.read_csv("club_type.csv")    
    df_legend = dict(zip(df_club_type["club"],df_club_type["club_type"]))
    df["club_type"] = df["club"].map(df_legend)

    df = df.replace({"club_type":     {"Elite": 1, "Tier1": 2,
                                  "Tier2": 3, "Tier3": 4, "Tier4": 5, "Tier5": 6,
                                  "Tier6": 7}})    
    
    df_new = pd.get_dummies(df, columns=["dominant_position", "preferred_foot"])
    df_legend = dict(zip(df["club"],df["club_type"]))

    df_main = df_new.reset_index()
    df_sub = df_new[["name","dob","traits"]].reset_index()
    df2_sub = pd.DataFrame(df_sub["traits"].str.split(";"))
    df2_sub = pd.DataFrame(df2_sub["traits"].values.tolist())
    df3_sub = pd.concat([df_sub[["index","name","dob"]],df2_sub], axis=1)
    df3_sub = df3_sub.melt(id_vars=["name","dob","index"])
    df3_sub["variable"] = 1
    df3_sub = df3_sub.drop_duplicates()
    df3_sub = df3_sub.dropna()
    df3_sub= df3_sub.pivot(index=["name","dob","index"], columns=["value"]).fillna(0).reset_index()
    df3_sub.columns = [' '.join(col).strip().replace("variable ","trait_").replace("-","_") for col in df3_sub.columns.values]
    df3_sub = df3_sub.drop(["name","dob"], axis=1)
    df_merged = df_main.merge(df3_sub, on="index")
    df_merged = df_merged.drop(["index"],axis=1)
    df_new = df_merged.drop(["dob","height_cm","weight_kg","club","potential","nationality","player_positions","talent","traits"], axis = 1)
    df_col = pd.read_csv("column_header.csv")
    df_col =  df_col.columns      
    for col in df_col :
        if col not in df_new.columns  :
            df_new[col] = 0 
            df_new = df_new[df_col] 

    with open(os.path.join(prod_deployment_path,"latestscore.txt"),"r") as f: 
        latestscore_prev = f.read()
    with open(os.path.join(prod_deployment_path,"trainedmodel.pkl"),"rb") as f:
        model = pickle.load(f)
    X_test = df_new.drop(["name","value_eur","wage_eur"], axis=1).values.reshape(-1,70)
    y_test = df["wage_eur"].values.reshape(-1,1).ravel()
    pred = model.predict(X_test)

    latestscore_now = model.score(X_test, y_test)    
    print(latestscore_prev)    
##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    if latestscore_now < float(latestscore_prev):
##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here        
        print("model drift has occured")
##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
        training.train_model()
        deployment.store_model_into_pickle(model_name="trainedmodel.pkl")
        reporting.score_model()
##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
        os.system("python run diagnostics.py")
    else :
        print("no model drift detected")
else :
    print("no new files")
##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model

