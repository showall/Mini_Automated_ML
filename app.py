from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import diagnostics
from scoring import score_model
import json
import os
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'
with open('config.json','r') as f:
    config = json.load(f) 
app.config['UPLOAD_FOLDER'] = config['test_data_path']
with open('config.json','r') as f:
    config = json.load(f) 
prod_deployment_path = config['prod_deployment_path'] 
test_data_path = config['test_data_path'] 
dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None

########################main page
@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template(
            'index.html',
    
           data=[{'gender': 'Gender'}, {'gender': 'female'}, {'gender': 'male'}],
            data1=[{'noc': 'Number of Children'}, {'noc': 0}, {'noc': 1}, {'noc': 2}, {'noc': 3}, {'noc': 4}, {'noc': 5}],
            data2=[{'smoke': 'Smoking Status'}, {'smoke': 'yes'}, {'smoke': 'no'}],
            data3=[{'region': "Region"}, {'region': "northeast"}, {'region': "northwest"},
                {'region': 'southeast'}, {'region': "southwest"}])

    if request.method == 'POST':   
        input_data = list(request.form.values())
        input_values = [x for x in input_data]
        arr_val = [np.array(input_values)]
        return arr_val 
     #   return redirect(url_for('mad'))


########################main page
@app.route('/predictone', methods = ['GET', 'POST'])
def predictone():
    if request.method == 'POST':        
        input_data = list(request.form.values())
        input_values = [x for x in input_data]
        arr_val = [np.array(input_values)]
        df_request = pd.DataFrame(zip(list(request.form),list(input_data))).set_index(0).T.reset_index()
        df_request =   df_request.drop(["index"], axis=1) 
        try: 
            df_request["traits"].iloc[0] =  ";".join(request.form.getlist('traits'))  
        except:
            pass
        try:
            df_request["talent"].iloc[0] = ";".join(request.form.getlist('talent'))      
        except:
            pass

        df_request = df_request.drop_duplicates()

        df_request["potential_out_of_overall"] = round(df_request["overall"].astype("float")/df_request["potential"].astype("float"),2)
        df_request["bmi"] = df_request.weight_kg.astype(float) *10000 / ( (df_request.height_cm.astype("float"))**2)
        df_request["bmi"] = round(df_request["bmi"].astype(float),2)

        df_club_type = pd.read_csv("club_type.csv")    
        df_legend = dict(zip(df_club_type["club"],df_club_type["club_type"]))
            
        df_request["club_type"] = df_request["club"].map(df_legend)

        df_request = df_request.replace({"club_type":     {"Elite": 1, "Tier1": 2,
                                    "Tier2": 3, "Tier3": 4, "Tier4": 5, "Tier5": 6,
                                    "Tier6": 7}})    

        df_col = pd.read_csv("column_header.csv")
        df_col =  df_col.columns          


        df_new = pd.get_dummies( df_request, columns=["dominant_position", "preferred_foot"])

        
        for col in ["name","dob","traits","talent","dominant_position", "preferred_foot"]:
            if col not in df_new.columns :
                if col == "traits":
                   df_new[col] = "ordinary"
                elif col == "dominant_position":
                   df_new[col] = "GK"
                elif col == "preferred_foot":
                   df_new[col] = "right"
                else :
                   df_new[col] = "user1"

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
        df_new = df_merged.drop(["dob","height_cm","weight_kg","club","potential","talent","traits"], axis = 1)         

        df_col = pd.read_csv("column_header.csv")

        df_col =  df_col.columns      
        for col in df_col :
            if col not in df_new.columns  :
                df_new[col] = 0 
        df_new = df_new[df_col] 
        
        X = df_new.drop(["name","value_eur","wage_eur"], axis=1).values.reshape(-1,70)
        with open(os.path.join(prod_deployment_path,"trainedmodel.pkl"),"rb") as f:
            model = pickle.load(f)
        result = model.predict(X)
        with open(os.path.join(prod_deployment_path,"latestscore.txt"),"r") as f: 
            latestscore_prev = f.read()
            latestscore_prev = float(latestscore_prev)
        return render_template('index.html', prediction_text=" The average monthly income is {} for the player of such calibre".format(result),
                            model_score = "{:.2f}".format(latestscore_prev))
                            # data=[{'gender': 'Gender'}, {'gender': 'female'}, {'gender': 'male'}],
                            # data1=[{'noc': 'Number of Children'}, {'noc': 0}, {'noc': 1}, {'noc': 2}, {'noc': 3},
                            #         {'noc': 4}, {'noc': 5}],
                            # data2=[{'smoke': 'Smoking Status'}, {'smoke': 'yes'}, {'smoke': 'no'}],
                            # data3=[{'region': "Region"}, {'region': "northeast"}, {'region': "northwest"},
                            #         {'region': 'southeast'}, {'region': "southwest"}])

    if request.method == 'GET':     
        return redirect(url_for('index'))  

#######################Mass upload
@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(config['test_data_path'], "testdata.csv"))
        #return 'file uploaded successfully'
        return redirect(url_for('prediction'))
    else :
        return render_template('upload.html')

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','GET'])
def prediction():        
    result = diagnostics.model_predictions("testdata.csv")
    return str(result)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    result = score_model(test_data_path)
    return str(result)

#######################Summary Statistics Endpoint
@app.route("/dataframe_summary", methods=['GET','OPTIONS'])
def stats():        
    result = diagnostics.dataframe_summary()
    return result

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnose():    
    
    result1 = diagnostics.missing_data_check()
    result2 = diagnostics.execution_time()
    result3 = diagnostics.outdated_packages_list()    
    dict = {"na_perc" : str(result1), 
            "exec_time" : str(result2),
              "outdated_dep" : str(result3),  
    }
    #result3 = diagnostics.outdated_packages_list()   
    #check timing and percent NA values
    #return 1
    return dict

if __name__ == "__main__":    
    app.run(port=8000, debug=True, threaded=True)
