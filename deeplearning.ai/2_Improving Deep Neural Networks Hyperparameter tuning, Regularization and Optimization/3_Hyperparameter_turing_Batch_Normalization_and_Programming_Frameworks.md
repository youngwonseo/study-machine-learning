# Week3 - Hypterparameter turing, Batch Normalization and Programmig Frameworks

## Turing proccess

##### Hyperparameters

* learning rate 1
* beta 2  (~ 0.9 good default)
* beta1(0.9), beta2(0.999), epsilon(10^-8) 3
* number of layers 3


* number of hidden units 2
* learning rate decay 3
* mini-batch size 2

 

##### Try random values : Don't use a grid

* ​

##### Coarse to fine





## Using an appropriate scale to pick hyperparameters

##### Picking hyperparameters at random

* ​

##### Appropriate scale for hyperparameters

* lambda = 0.0001 ... , 1
* linear scale, log scale
  * r = -4 * np.random.rand()
  * lambda = 10^r

##### Hyperparameters for exponentially weighted averages

* beta = 0.9 ... 0.999



## Hyperparameters turing in pratice: Pandas vs. Caviar

* Babysitting one model
* Training many models in parallel



## Normalizing activations in a network



 ## Fitting Batch Norm into a neural network

## Why does Batch Norm work?

## Batch Norm at test time

