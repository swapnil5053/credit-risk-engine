import json
from main import _load_config, CONFIG_PATH
from src.data_loader import load_dataset

def main():
    config = _load_config(CONFIG_PATH)
    X, y = load_dataset(config)
    
    # Grab the very first row
    first_customer = X.iloc[0].to_dict()
    
    # Print as a formatted JSON string (using default=str to handle any Pandas data types)
    print(json.dumps(first_customer, indent=2, default=str))

if __name__ == "__main__":
    main()
