import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.tool import save_meta_data
from utils.dataset import CustomDataset
from utils.step import step
from nn_model import DeepFM

### Initialize GPU
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

### Parameter
SAVE_PATH = "./saved_model"
RESULT_PATH = "result.csv"
RANDOM_SEED = 999

### Hyper Parameter
VALID_RATIO = 0.1
BATCH_SIZE = 2000
INTI_LR = 1e-1
MIN_LR = 1e-2
TOTAL_EPOCHS = 5
OVERFITTING_LIMIT = 5
EMBEDDING_DIM = 4
HIDDEN_DIM = 4
DROPOUT_RATIO = 0.3


def train_model(
    catalog_nums,
    train_loader,
    valid_loader,
):
    ### 1. Load Model
    model = DeepFM(
        device=DEVICE,
        catalog_nums=catalog_nums,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        bucket_nums=4,
        hidden_layer_nums=4,
        dropout_ratio=DROPOUT_RATIO
    )
    model.to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=INTI_LR, 
        betas=(0.99, 0.999),
        weight_decay=1e-5,
    )
    criterion = nn.BCELoss().to(DEVICE)
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     T_max=5, 
    #     eta_min=MIN_LR
    # )

    ### 2. Start Train 
    _best_loss = 1e3
    _not_updated_step = 0

    total_train_outputs = []
    total_valid_outputs = []

    for epoch in range(1, TOTAL_EPOCHS+1):
        print(f"###### Train : ({epoch}/{TOTAL_EPOCHS}) ######")
        train_output = step(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            is_train=True,
            device=DEVICE,
            logging_nums=5
        )
        total_train_outputs.append(train_output)
        
        print(f"###### Validation : ({epoch}/{TOTAL_EPOCHS}) ######")
        valid_output = step(
            model=model,
            data_loader=valid_loader,
            optimizer=optimizer,
            criterion=criterion,
            is_train=False,
            device=DEVICE,
            logging_nums=2
        )
        total_valid_outputs.append(valid_output)
        
        # scheduler.step()

        ## Save
        if valid_output["loss"] < _best_loss:
            _not_updated_step = 0
            _best_loss = valid_output["loss"]    
            if not os.path.isdir(SAVE_PATH):
                os.mkdir(SAVE_PATH)     

            full_save_path = os.path.join(SAVE_PATH, "weight.pth")
            print("Saving Model..")
            torch.save(model, full_save_path)    
        else:
            _not_updated_step += 1
        
        if _not_updated_step >= OVERFITTING_LIMIT:
            print("End Training..")
            break

    return total_train_outputs, total_valid_outputs


if __name__ == "__main__":

    ### Load Data
    data = pd.read_csv('train.csv', header=None)

    ## 1. Split Dataset to Train & Valid
    valid_data_1 = data[data[0] == 0].sample(frac = VALID_RATIO, random_state=RANDOM_SEED)
    valid_data_2 = data[data[0] == 1].sample(frac = VALID_RATIO, random_state=RANDOM_SEED)
    valid_df = pd.concat([valid_data_1, valid_data_2])
    valid_df = valid_df.sample(frac=1)

    valid_indices = set(valid_df.index)

    train_df = data[~data.index.isin(valid_indices)]

    ## Train Dataset
    train_dataset = CustomDataset(
        df = train_df,
        dataset_type = "train"
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size = BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    categorical_layers, catalog_nums, numerical_meta = train_dataset.export_data()

    ## Save Meta Data
    save_meta_data(
        categorical_layers = categorical_layers,
        catalog_nums = catalog_nums,
        numerical_meta = numerical_meta,
        save_path = SAVE_PATH,
    )

    ## Valid Dataset
    valid_dataset = CustomDataset(
        df = valid_df,
        dataset_type = "valid",
        categorical_layers = categorical_layers,
        catalog_nums = catalog_nums,
        numerical_meta = numerical_meta
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size = BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    ## 2. Load Model
    train_output, valid_output = train_model(
        catalog_nums=catalog_nums,
        train_loader=train_loader,
        valid_loader=valid_loader
    )
    
    df = pd.DataFrame(train_output, columns=["loss", "accuracy", "precision", "recall", "f1", "auc"])
    df.to_csv(RESULT_PATH)     
