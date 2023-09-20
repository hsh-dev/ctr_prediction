#!/bin/bash

python inference.py --save_path ./lr_0.01_emb_4_drop_0.3
python inference.py --save_path ./lr_0.01_emb_8_drop_0.3
python inference.py --save_path ./lr_0.01_emb_16_drop_0.3
python inference.py --save_path ./lr_0.01_emb_32_drop_0.3
python inference.py --save_path ./lr_0.01_emb_64_drop_0.3

## drop out
python inference.py --save_path ./lr_0.01_emb_16_drop_0.1
python inference.py --save_path ./lr_0.01_emb_16_drop_0.5
python inference.py --save_path ./lr_0.01_emb_16_drop_0.7
python inference.py --save_path ./lr_0.01_emb_16_drop_0.9

## Inverse Focal Loss
python inference.py --save_path ./lr_0.01_emb_4_drop_0.3_gamma_3_inv
python inference.py --save_path ./lr_0.01_emb_8_drop_0.3_gamma_3_inv
python inference.py --save_path ./lr_0.01_emb_16_drop_0.3_gamma_3_inv
python inference.py --save_path ./lr_0.01_emb_32_drop_0.3_gamma_3_inv
python inference.py --save_path ./lr_0.01_emb_64_drop_0.3_gamma_3_inv

## Gamma
python inference.py --save_path ./lr_0.01_emb_16_drop_0.3_gamma_1_inv
python inference.py --save_path ./lr_0.01_emb_16_drop_0.3_gamma_2_inv
python inference.py --save_path ./lr_0.01_emb_16_drop_0.3_gamma_4_inv
python inference.py --save_path ./lr_0.01_emb_16_drop_0.3_gamma_5_inv
