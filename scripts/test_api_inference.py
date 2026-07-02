import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import requests
import json
from main import _load_config, CONFIG_PATH
from src.data_loader import load_dataset
import pandas as pd

def main():
    config = _load_config(CONFIG_PATH)
    X, y = load_dataset(config)
    
    # Grab the very first customer
    first_customer = X.iloc[0].to_dict()
    
    import math
    for k, v in first_customer.items():
        if isinstance(v, float) and math.isnan(v):
            first_customer[k] = None
    
    # Send it to our running FastAPI server
    response = requests.post("http://127.0.0.1:8000/predict", json=first_customer)
    
    print("\n--- API RESPONSE ---")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    main()
