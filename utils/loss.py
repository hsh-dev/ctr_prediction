import torch

def inverse_focal_loss(gamma = 3):
    MIN = 1e-7
    MAX = 1 - MIN
    def loss_func(y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=MIN, max=MAX)

        ## Term for positive label
        pos = (-y_true) * torch.float_power((1 + y_pred), gamma) * torch.log(y_pred) 
        
        ## Term for negative label
        neg = (-1) * (1 - y_true) * torch.float_power((2 - y_pred), gamma) * torch.log(1 - y_pred)
    
        loss = torch.mean(pos + neg)
        return loss
    
    return loss_func
    