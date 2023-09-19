import json
import os


def save_meta_data(
    categorical_layers,
    catalog_nums,
    numerical_meta,
    save_path: str = "./saved_model"
):
    meta_data = {
        "categorical_layers": categorical_layers,
        "catalog_nums": catalog_nums,
        "numerical_meta": numerical_meta
    }

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    meta_data_path = os.path.join(save_path, "meta_data.json")
    
    with open(meta_data_path, 'w') as f:
        json.dump(meta_data, f, indent=4)
        
def load_meta_data(
    data_path  
):
    meta_data_path = os.path.join(data_path, "meta_data.json")
    
    with open(meta_data_path, "r") as f:
        meta_data = json.load(f)
        
    return meta_data
