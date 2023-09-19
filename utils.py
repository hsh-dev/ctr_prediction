import json
import os
from sklearn.metrics import confusion_matrix


def calculate_confusion_mat(
    logits, 
    labels,
    theshold: float = 0.5
):
    def _pred(x, theshold):
        if x >= theshold:
            return 1
        else:
            return 0
        
    y_pred = [_pred(i, theshold) for i in logits]
    y_true = labels
    
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred 
    ).ravel()
    
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    
def calculate_score(
    total_cm : dict[str, int]
):
    try:
        precision = total_cm['tp'] / (total_cm['fp']+ total_cm['tp'])
    except ZeroDivisionError:
        precision = 0            
    try:
        recall = total_cm['tp'] / (total_cm['tp'] + total_cm['fn'])
    except ZeroDivisionError:
        recall = 0            
    try:
        accuracy = (total_cm['tp'] + total_cm['tn']) / (total_cm['tp'] + total_cm['fn'] + total_cm['fp'] + total_cm['tn'])
    except ZeroDivisionError:
        accuracy = 0            
    try:
        F1 = 2 *( (precision * recall) / (precision + recall) )
    except ZeroDivisionError:
        F1 = 0        
    return accuracy, precision, recall, F1

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
