import time
import sys

import numpy as np
from utils.metrics import (
    calculate_confusion_mat, 
    calculate_score
)
from sklearn.metrics import roc_auc_score 

def step(
    model,
    data_loader,
    optimizer,
    criterion,
    is_train: bool = True,
    device: str = "cpu",
    logging_nums: int = 10
):      
    if is_train:
        model.train()
    else:
        model.eval()

    step = 1
    steps = len(data_loader)
    logging_step_size = steps // logging_nums
    
    ## loss
    total_loss = []
    temp_loss = []
    
    ## pred, label
    total_logits = []
    total_labels = []
    
    ## score
    total_cm = {
        "tp": 0, "fp": 0, "fn": 0, "tn": 0
    }
    
    prev_time = time.time()
    for data in iter(data_loader):
        label, numeric_values, categoric_values, numeric_indicators, categoric_indicators = data

        label = label.float().to(device)
        numeric_values = numeric_values.to(device)
        categoric_values = categoric_values.to(device)
        numeric_indicators = numeric_indicators.to(device)
        categoric_indicators = categoric_indicators.to(device)
        
        if is_train:
            optimizer.zero_grad()

        logit = model(
            numeric_values,
            categoric_values,
            numeric_indicators,
            categoric_indicators
        )                
        
        loss = criterion(logit.squeeze(1), label)
        
        if is_train:
            loss.backward()
            optimizer.step()

        ## Calculate Loss, Score
        temp_loss.append(loss.item())
        total_loss.append(loss.item())
        
        ## Score
        logits = list(logit.cpu().squeeze(1).detach().numpy())
        labels = list(label.cpu().detach().numpy())
        
        total_logits.extend(logits)
        total_labels.extend(labels)
        
        temp_cm = calculate_confusion_mat(
            logits=logits,
            labels=labels,
            theshold=0.5
        )
        for _k, _v in temp_cm.items():
            total_cm[_k] += _v
        
        ## Logging
        if step % logging_step_size == 0:
            current_loss = np.mean(temp_loss)
            timedelta = time.time() - prev_time
            accuracy, precision, recall, f1 = calculate_score(
                total_cm=temp_cm
            )
            
            print(
                f"[ Step : ({step}/{steps}) "
                f"| Loss : {current_loss:.4f} "
                f"| Time : {timedelta:.4f}s]"
            )
            print(
                f"[ Accuracy : {accuracy:.6f} "
                f"| Precision : {precision:.6f} "
                f"| Recall : {recall:.6f} "
                f"| F1 : {f1:.6f}]"
            )
            sys.stdout.flush()
            
            temp_loss.clear()
            prev_time = time.time()

        step = step+1
    
    ## For epoch summary
    _total_loss = np.mean(total_loss)
    total_accuracy, total_precision, total_recall, total_f1 = calculate_score(
        total_cm=total_cm
    )
    total_auc = roc_auc_score(total_labels, total_logits)
    print("[ Total Result ]")
    print(
        f"[ Loss : {_total_loss:.4f} "
        f"| Accuracy : {total_accuracy:.6f} "
        f"| Precision : {total_precision:.6f} "
        f"| Recall : {total_recall:.6f} "
        f"| F1 : {total_f1:.6f} "
        f"| AUC : {total_auc:.6f} ]"
    )
    print(f"[ TP : {total_cm['tp']} | FN : {total_cm['fn']} | TN : {total_cm['tn']} | FP : {total_cm['fp']} ]")
    
    return {
        "loss": _total_loss,
        "accuracy": total_accuracy,
        "precision": total_precision,
        "recall": total_recall,
        "f1": total_f1,
        "auc": total_auc,
        "cm": total_cm,
        "logits": total_logits,
        "labels": total_labels
    }
