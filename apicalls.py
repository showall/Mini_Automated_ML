import requests
import os
import pandas as pd
import json
from datetime import datetime

#Specify a URL that resolves to your workspace
URL = "http://192.168.0.166:"
with open('config.json','r') as f:
    config = json.load(f) 
dataset_csv_path = config['output_folder_path'] 
output_model_path = os.path.join(config['output_model_path'])
dataset_file_location = os.path.join(config['test_data_path'])

#Call each API endpoint and store the responses
response1 = requests.get(f"{URL}8000/prediction").content
response2 = requests.get(f"{URL}8000/scoring").content
response3 = requests.get(f"{URL}8000/dataframe_summary").content
response4 = requests.get(f"{URL}8000/diagnostics").content

#combine all API responses
responses = [response1,response2, response3, response4] 
#write the responses to your workspace
date = datetime.now().strftime("%Y%m%d_%H%M%S")

with open(os.path.join(output_model_path,f"apireturns_{date}.txt"),"wb") as f:
    for response in responses :
        f.write(response)
        f.write('\n'.encode('utf-8'))



