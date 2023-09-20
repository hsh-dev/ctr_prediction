#!/bin/bash

## embedding dim
python train.py --save_path ./lr_0.01_emb_4_drop_0.3 --lr 0.01 --emb_dim 4 --hidden_dim 4
python train.py --save_path ./lr_0.01_emb_8_drop_0.3 --lr 0.01 --emb_dim 8 --hidden_dim 8
python train.py --save_path ./lr_0.01_emb_16_drop_0.3 --lr 0.01 --emb_dim 16 --hidden_dim 16
python train.py --save_path ./lr_0.01_emb_32_drop_0.3 --lr 0.01 --emb_dim 32 --hidden_dim 32
python train.py --save_path ./lr_0.01_emb_64_drop_0.3 --lr 0.01 --emb_dim 64 --hidden_dim 64

## drop out
python train.py --save_path ./lr_0.01_emb_16_drop_0.1 --lr 0.01 --emb_dim 16 --hidden_dim 16 --dropout 0.1
python train.py --save_path ./lr_0.01_emb_16_drop_0.5 --lr 0.01 --emb_dim 16 --hidden_dim 16 --dropout 0.5
python train.py --save_path ./lr_0.01_emb_16_drop_0.7 --lr 0.01 --emb_dim 16 --hidden_dim 16 --dropout 0.7
python train.py --save_path ./lr_0.01_emb_16_drop_0.9 --lr 0.01 --emb_dim 16 --hidden_dim 16 --dropout 0.9

## Inverse Focal Loss
python train.py --save_path ./lr_0.01_emb_4_drop_0.3_gamma_3_inv --lr 0.01 --emb_dim 4 --hidden_dim 4 --gamma 3
python train.py --save_path ./lr_0.01_emb_8_drop_0.3_gamma_3_inv --lr 0.01 --emb_dim 8 --hidden_dim 8 --gamma 3
python train.py --save_path ./lr_0.01_emb_16_drop_0.3_gamma_3_inv --lr 0.01 --emb_dim 16 --hidden_dim 16 --gamma 3
python train.py --save_path ./lr_0.01_emb_32_drop_0.3_gamma_3_inv --lr 0.01 --emb_dim 32 --hidden_dim 32 --gamma 3
python train.py --save_path ./lr_0.01_emb_64_drop_0.3_gamma_3_inv --lr 0.01 --emb_dim 64 --hidden_dim 64 --gamma 3

## Gamma
python train.py --save_path ./lr_0.01_emb_16_drop_0.3_gamma_1_inv --lr 0.01 --emb_dim 16 --hidden_dim 16 --gamma 1
python train.py --save_path ./lr_0.01_emb_16_drop_0.3_gamma_2_inv --lr 0.01 --emb_dim 16 --hidden_dim 16 --gamma 2
python train.py --save_path ./lr_0.01_emb_16_drop_0.3_gamma_4_inv --lr 0.01 --emb_dim 16 --hidden_dim 16 --gamma 4
python train.py --save_path ./lr_0.01_emb_16_drop_0.3_gamma_5_inv --lr 0.01 --emb_dim 16 --hidden_dim 16 --gamma 5
