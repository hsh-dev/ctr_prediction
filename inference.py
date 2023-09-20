import argparse
import os
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.tool import load_meta_data
from utils.dataset import CustomDataset
from utils.step import step
from utils.metrics import calculate_ece

parser = argparse.ArgumentParser()
parser.add_argument("--save_path")

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SAVE_PATH = "./saved_model"
TEST_RESULT_FILE = "test_result.csv"


if __name__ == "__main__":
    ## Parse Args
    args = parser.parse_args()
    if args.save_path is not None:
        SAVE_PATH = args.save_path
    
    meta_data = load_meta_data(SAVE_PATH)
    categorical_layers = meta_data["categorical_layers"]
    catalog_nums = meta_data["catalog_nums"]
    numerical_meta = meta_data["numerical_meta"]

    test_data = pd.read_csv('test.csv', header=None)


    test_dataset = CustomDataset(
        df = test_data,
        dataset_type = "test",
        categorical_layers = categorical_layers,
        catalog_nums = catalog_nums,
        numerical_meta = numerical_meta
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size = 2000, 
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = torch.load(SAVE_PATH + "/weight.pth")

    model.to(DEVICE)
    criterion = nn.BCELoss().to(DEVICE)
    test_result = step(
        model=model,
        data_loader=test_loader,
        optimizer=None,
        criterion=criterion,
        is_train=False,
        device=DEVICE,
        logging_nums=5
    )

    columns = [
        "loss", "accuracy", "precision", "recall", 
        "f1", "auc", "cm", "logits", "labels"
    ]
    output = {}
    for col in columns:
        output[col] = [test_result[col]]

    ece = calculate_ece(
        logits=test_result["logits"], 
        labels=test_result["labels"]
    )
    output["ece"] = [ece]

    test_result_path = os.path.join(SAVE_PATH, TEST_RESULT_FILE)
    pd.DataFrame(
        output, 
    ).to_csv(test_result_path, index=False)
