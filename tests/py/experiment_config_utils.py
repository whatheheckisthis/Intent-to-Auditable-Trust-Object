import json

def load_config(file_path):
    with open(file_path,'r') as f:
        return json.load(f)

def save_config(cfg, file_path):
    with open(file_path,'w') as f:
        json.dump(cfg, f, indent=2)
