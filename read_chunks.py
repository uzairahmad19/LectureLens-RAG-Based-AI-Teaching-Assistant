import requests
import json,os
import pandas as pd
import joblib

# create embedding for each chunk and add it to the json file
def create_embeddings(text_list):
    r = requests.post('http://localhost:11434/api/embed', json={
        'model': 'bge-m3', 'input': text_list
    })
    embedding = r.json()['embeddings']
    return embedding

jsons = os.listdir('transcripts') # List all json files in the transcripts directory
my_dict = []
chunk_id = 0

for json_file in jsons:
    with open(f'transcripts/{json_file}', 'r') as f:
        data = json.load(f)
    embeddings = create_embeddings([chunk['text'] for chunk in data['chunk']]) # Create embeddings for each chunk of text
    for i, chunk in enumerate(data['chunk']):
        chunk['embedding'] = embeddings[i]
        chunk['chunk_id'] = chunk_id
        chunk_id += 1
        my_dict.append(chunk)

df = pd.DataFrame.from_records(my_dict) # Create a DataFrame from the list of dictionaries

joblib.dump(df, 'chunks_with_embeddings.joblib') # Save the DataFrame as a joblib file for later use