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
def merge_multiple_dataframe(input_folder_path, train=False):
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
   # df1 = pd.get_dummies(df["talent"])
   # df2 = pd.get_dummies(df["traits"])
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

#   df_legend = dict(zip(df["club"],df["club_type"]))
#    print(df_legend)

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
    df_new[0:1].to_csv("column_header.csv",index=False)     
    
    if train == False:
        df_col = pd.read_csv("column_header.csv")
        df_col =  df_col.columns      
        for col in df_col :
            if col not in df_new.columns  :
                df_new[col] = 0 
                df_new = df_new[df_col] 

    return df_new


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