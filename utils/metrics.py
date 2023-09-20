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

def calculate_ece(
    logits,
    labels,
    threshold: float = 0.5,
):
    m = 0.1
    bin_dict = {i : {"logits": [], "labels": []} for i in range(10)}
    bin_min = [0.1*i  for i in range(10)]

    for _logit, _label in zip(logits, labels):
        
        bin_idx = None
        for idx, _bin in enumerate(bin_min):
            if _logit >= _bin:
                continue
            elif _logit < _bin:
                bin_idx = idx - 1
                break
        
        if bin_idx is None:
            bin_idx = 9
        
        bin_dict[bin_idx]["logits"].append(_logit)
        bin_dict[bin_idx]["labels"].append(_label)

    diff_list = []
    total_counts = 0
    for i in range(10):
        _logits = bin_dict[i]["logits"]
        _labels = bin_dict[i]["labels"]
        
        if len(_logits) == 0:
            continue
        
        correct = 0
        for _logit, _label in zip(_logits, _labels):
            if (_logit >= threshold and _label == 1) or\
                (_logit < threshold and _label == 0):
                correct += 1

        counts = len(_logits)
        total_counts += counts
        accuracy = correct / counts
        average = sum(_logits) / counts
        
        diff_list.append(abs(accuracy-average) * counts)

    ece = sum(diff_list) / len(diff_list) / total_counts

    return ece