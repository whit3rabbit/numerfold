import numpy as np
import pandas as pd
from datetime import datetime
import json
import os

def make_json_serializable(obj):
    """
    Recursively convert a nested structure containing numpy arrays, pandas DataFrames,
    and other non-serializable types into JSON-serializable types.
    
    Parameters:
        obj: The object to make serializable
        
    Returns:
        A JSON-serializable version of the object
    """
    
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.DataFrame):
        # Convert DataFrame to a dict of lists
        return {
            'columns': obj.columns.tolist(),
            'data': obj.values.tolist()
        }
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'to_dict'):
        # Handle objects with to_dict method
        return make_json_serializable(obj.to_dict())
    elif hasattr(obj, '__dict__'):
        # Handle custom objects by converting their __dict__
        return make_json_serializable(obj.__dict__)
    else:
        # Return the object as is if it's likely serializable
        # or convert to string as fallback
        try:
            # Test if it's JSON serializable
            import json
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

def save_results_to_json(results, filename='results.json'):
    """
    Save results dictionary to a JSON file, handling non-serializable types.
    
    Parameters:
        results: Dictionary of results to save
        filename: Output JSON filename
        
    Returns:
        str: Path to saved file
    """
   
    # Make results serializable
    serializable_results = make_json_serializable(results)
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {os.path.abspath(filename)}")
    return os.path.abspath(filename)