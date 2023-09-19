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
        y_true, y_pred, labels=[0, 1]
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
