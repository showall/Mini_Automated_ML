import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
model_path = os.path.join(config['output_model_path']) 
#############Function for data ingestion
def merge_multiple_dataframe(input_folder_path):
    #check for datasets, compile them together, and write to an output file
 #   df = pd.DataFrame(columns=["corporation","lastmonth_activity","lastyear_activity","number_of_employees","exited"])

    df = pd.DataFrame(columns= ["name","age","dob",
     "height_cm","weight_kg",
     "nationality","club","overall","potential",	
     "value_eur", "wage_eur","player_positions",
     "preferred_foot","international_reputation",
     "weak_foot","skill_moves", "work_rate", "versatility",
	"release_clause","talent", 
    "dominant_position", "traits"] )

    for file in os.listdir(os.path.join(input_folder_path)):
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
#    print(df_legend)
    df_new = df_new.drop(["dob","height_cm","weight_kg","club","potential","nationality","player_positions","talent","traits"], axis = 1)
   # print(df_new.head(5))

    return df_new
    p = Path(output_folder_path)
    p.mkdir(exist_ok=True)
    df_new.to_csv(os.path.join(p ,"finaldata.csv"),index=False)
    with open(os.path.join(p ,"ingestedfiles.txt"),"w") as f:
        files = []
        for file in os.listdir(os.path.join(input_folder_path)):
            files.append(file)
        f.write(str(files))
           # f.write(str("\n"))

if __name__ == '__main__':
    df_new = merge_multiple_dataframe(input_folder_path)
    p = Path(output_folder_path)
    p.mkdir(exist_ok=True)
    df_new.to_csv(os.path.join(p ,"finaldata.csv"),index=False)
    with open(os.path.join(p ,"ingestedfiles.txt"),"w") as f:
        files = []
        for file in os.listdir(os.path.join(input_folder_path)):
            files.append(file)
        f.write(str(files))
           # f.write(str("\n"))