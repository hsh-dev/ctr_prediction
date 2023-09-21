### Deepest Quest #3

### 데이터 설명
criteo-small 데이터셋
- https://www.kaggle.com/datasets/leonerd/criteo-small

### 학습 방법

#### 1. BCE loss로 학습할 때
learning rate, embedding dimension, hidden dimension, dropout ratio를 조절하는 경우
```shell
python train.py --save_path ./lr_0.01_emb_4_drop_0.3 --lr 0.01 --emb_dim 4 --hidden_dim 4 --dropout 0.3
```

#### 2. Inverse Focal loss로 학습할 때
```shell
python train.py --save_path ./lr_0.01_emb_4_drop_0.3_gamma_3_inv --lr 0.01 --emb_dim 4 --hidden_dim 4 --gamma 3
```

### 테스트 방법

테스트하고자 하는 모델이 저장된 경로를 입력한다.
```shell
python inference.py --save_path ./lr_0.01_emb_16_drop_0.3_gamma_1_inv
```