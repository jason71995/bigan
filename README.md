# Implementation of BiGAN on Keras and Tensorflow 2.0

## Introduction
Implementation of [(2016) Adversarial Feature Learning.](https://arxiv.org/abs/1605.09782)

ATTENTION:
The architectures and the hyperparameters in this project are not exactly same as the paper.

## Environment

```
python==3.6
tensorflow==2.0
```

## Results

Results at 200k iterations.

|Name|X|G(E(X))|G(Z)|
|:---:|:---:|:---:|:---:|
|MNIST|![alt text](https://imgur.com/kbG4sXX.png "X") |![alt text](https://imgur.com/OgkEJDQ.png "G(E(X))")|![alt text](https://imgur.com/bhkoNKX.png "G(Z)")|
|CIFAR10|![alt text](https://imgur.com/VjhuQmP.png "X") |![alt text](https://imgur.com/7XYL7Ma.png "G(E(X))")|![alt text](https://imgur.com/3tH8y4n.png "G(Z)")|