## 디피스트 과제 - 홍성훈

### 1. CTR Prediction

#### 1.1. DeepFM 논문 요약

**DeepFM의 Contribution**

DeepFM은 기존 CTR 예측 모델로 사용되었던 FNN, PNN, Wide&Deep 모델의 한계를 개선하였고, AUC와 logloss 성능 또한 SOTA를 달성하였다. Factorization Machine(FM) 이후, high-order feature 끼리 pair-wise interaction을 계산할 때 엄청나게 sparse한 matrix를 만드는 문제가 개선되었다. FNN의 경우 high-order feature에 대한 latent vector들의 inner product를 계산함으로써 high-order feature에 대한 학습은 가능하지만 FM pretraining이 필요하다는 단점이 있다. PNN의 경우 pretraining 없이 학습을 할 수 있도록 FM layer와 유사한 역할을 하는 product layer를 모델에 포함시켰지만 low-order feature에 대한 학습은 불가능했다. Wide&Deep의 경우 high-order feature와 low-order feature를 모두 학습할 수 있는 구조를 제안했지만, low-order feature에 대해 feature engineering이 필요했다. 이 논문에서 제안한 DeepFM의 경우 Wide&Deep 모델과 달리 feature engineering이 필요없도록 low-order feature 또한 high-order feature와 동일한 latent vector space에 매핑하여 표현한다.

**모델 아키텍쳐**

DeepFM의 경우 FM component와 deep component로 구성된다. 모델의 input의 경우 user, item feature를 concat하여 만든 sparse feature vector이다. 최종 output은 0과 1사이의 CTR 예측값이다. 이를 식으로 표현하면 (1)과 같다.
$$
\hat{y} = sigmoid(y_{FM}+y_{DNN})
$$
**FM Component**

FM component의 경우 order-1 feature들의 linear combination과 order-2 feature들의 interaction을 나타낸다. 이를 식으로 표현하면 (2)와 같다.
$$
y_{FM} = <w, x> + \sum_{i=1}^{d} \sum_{j=i+1}^{d} <V_{i}, V_{j}>x_{i} \cdot x_{j}
$$
order-1 feature의 linear combination은 앞부분에 해당된다. order-2 feature의 경우 뒷부분에 해당된다. V_i와 V_j는 각각 feature i, j에 대한 latent vector를 의미한다. V_i와 V_j에 대한 inner product를 feature i와 j가 모두 input에 존재하는 경우, 즉 x_i = 1,  x_j = 1 일 때만 계산한다. 두 feature간의 상관관계를 나타내는 order-2 feature를 계산함으로써 조금 더 복잡한 인풋을 모델링할 수 있다.

**Deep Component**

Deep component의 경우 FM component에서 사용하는 latent vector를 따로 pretrain 하여 사용하지 않고 그대로 사용한다. Latent vector들을 dense layer에 여러 번 통과시키는 구조로 되어 있다. dense layer의 총 개수는 H일 때 식 (3)과 같이 표현된다.
$$
a^{0} = [V_1, V_2, ..., V_d]\\
a^{l+1} = \sigma(W^{l}a^{l} + b^{l})\\
y_{DNN} = \sigma(W^{|H|+1} \cdot a^{H} + b^{|H|+1} )
$$
Deep component에서는 FM component과 달리 activation function을 통해 모델의 복잡도를 올릴 수 있었고, 이를 통해 order-1 feature나 order-2 feature로 표현하지 못하는 feature interaction 정보를 표현할 수 있었다.

**성능**

이 논문에서는 Criteo dataset (오픈 데이터셋) 과 Company dataset (비공개)를 사용하여 모델들을 평가하였다. 지표의 경우 AUC와 cross entropy loss로 측정하였다. 두 데이터셋에 대해서 모두 DeepFM은 SOTA를 달성하였다. 



#### 1.2. Criteo Dataset 학습 결과

1.2.1. Preprocessing

데이터셋의 경우 criteo-small를 사용하였다. 총 100만개의 user&item feature와 label (0/1) 데이터가 있었고, 이 중 90%를 train dataset, 나머지 10%를 test dataset으로 분리하였습니다. 분리할 때는 label의 비율 또한 유지되도록 random sampling 하였다. 13개의 continuous feature의 경우 numerical 값을 여러 개의 bag로 분류해 각각에 대해 meta embedding을 weight sum 방식을 사용하였다 (참고 2). 나머지 26개의 categorical feature의 경우 train dataset에서 feature 별로 나타난 unique categorical value를 모두 각각의 embedding vector로 만들었다.



1.2.2. Train

Train dataset의 10%를 validation에 사용하였다.

사용한 loss : BCE loss

측정 지표 : Loss, Recall, F1 (Recall를 선택한 이유는 CTR데이터 상 유저가 클릭한 데이터는 클릭하지 않은 데이터에 비해서 매우 적고, 유저가 실제로 클릭한 경우를 모델이 얼마나 잘 예측하는지가 CTR예측 추천 모델의 지표라 생각함)

하이퍼파라미터 튜닝 : dropout ratio, learning rate, embedding dimension





1.2.3. Result







---

### 2. Calibration

#### 2.1. Calibration 논문 (Inverse Focal Loss) 요약

**논문의 실험 내용**

이 논문에서는 딥러닝 모델에 대한 calibration 방법들을 실험적으로 재분석하였다.

Calibration을 하는 방법으로 모델에 대해 regularization을 주는 방법이 있다. Overfitting을 방지하기 위해 제안된 label smoothing, class imbalance 문제를 해결하기 위해 제안된 focal loss, L1, L2 normalization이 이 방법에 속한다. Cross entropy loss로 학습한 것과 비교하였을 때 label smoothing, focal loss, L1 norm 은 모두 Expected Calibration Error (ECE) 측면에서 좋은 결과를 냈다. 따라서 regularization 방법론이 calibration에 효과는 있다는 것이 증명되었다.

하지만 이런 regularization 방법론에 temperature scaling (TS) 나 histogram binning (HB) 와 같은 post-hoc 방식을 추가로 적용하였을 때는 ECE값이 regularization을 적용하지 않았을 때 보다 높게 나왔다. Regularization 방법론은 calibration에 효과는 있지만, post-hoc 방식에 의한 calibration의 잠재적인 개선폭은 오히려 감소시킨다는 결과이다.

이 논문에서 주장하는 것은 regularization 방법론에 의해 easy sample에 대한 confidence에 페널티가 가해지고, 이로인해 easy sample과 hard sample을 잘 구분하기 힘들다고 한다. CE loss에 대한 gradient을 구했을 때 (4)과 같다. 이때 confidence의 값이 작은 hard sample일수록 페널티가 더 커짐을 알 수 있다. 
$$
\frac{\partial \mathcal{L_{ce}}(y, f^{\theta}(x))}{\partial \theta}= - \frac{1}{f^{\theta}_{y}(x)}\nabla_{\theta}f^{\theta}_{y}(x)
$$
Focal loss의 경우 gradient를 계산하게 되면 confidence의 값이 클 때 CE loss에 비해 절댓값이 더 작다. confidence가 높은 easy sample에 대해서 CE loss보다 덜 페널티를 준다고 볼 수 있다.

이 논문에서 제안한 loss는 inverse focal loss로 식은 (5)와 같다.
$$
\mathcal{L}_{\bar{f}} =-(1+f^{\theta}_{y})^{\bar{\gamma}}log f^{\theta}_{y}
$$
Inverse focal loss의 경우 gradient를 계산했을 때 CE loss에 비해 confidence가 높은 easy sample에 대해서 더 큰 페널티를 줄 수 있다. 이는 easy sample에 대해서 overconfidence하게 만들고 easy sample과 hard sample을 더 구분하게 된다.

**실험 결과**

제안된 inverse focal loss의 경우 post-hoc을 사용하지 않았을 때 CE loss나 focal loss에 비해 높은 ECE 값이 나왔다. 하지만 TS나 HB와 같은 post-hoc 기법을 사용하였을 때inverse focal loss의 ECE값이 대부분 더 낮게 나왔다. 데이터셋에 따라 결과가 상이하였는데, 이 논문에서 주장하기로는 SVHN과 같이 라벨을 구분하기 쉬운 데이터셋에 대해서는 CE loss가 성능이 가장 좋았고 CIFAR-10이나 CIFAR-100과 같이 복잡하고 어려운 데이터셋에 대해서는 inverse focal loss의 성능이 좋았다고 한다.

**해당 모델의 핵심 아이디어**

위 논문의 아이디어는 confidence가 높게 나온 easy sample에 페널티를 더 많이 주면 calibration 성능이 좋아진다고 생각한 것이다. Inverse focal loss를 사용한다면, 학습 초기에는 easy sample에 대해서 gradient 값이 크기 때문에 easy sample들이 먼저 학습이 될 것이다. 이후 easy sample들이 overconfident 해지면, 자연스럽게 전체 loss중에 hard sample에 대한 loss 값의 비중이 높아져서 hard sample들도 학습이 될 것 같다. 다만, inverse focal loss만으로는 calibration을 최적화하기 힘들고 TS나 HB와 같은 추가적인 작업이 필요한 점이 아쉽다.



#### 2.2. Criteo Dataset Calibration 적용 결과





---

3. Bonus







---

### 4. 참고자료

1. criteo-small : 

2. An Embedding Learning Framework for Numerical Features in CTR Prediction (https://arxiv.org/pdf/2012.08986.pdf)

