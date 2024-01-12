import json 

def load_json(filepath):
    return json.load(open(filepath, 'r'))