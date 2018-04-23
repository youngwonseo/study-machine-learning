# Week2 - Optimization

## Mini-batch gradient descent





## Understading mini-batch gradient descent

##### Stochastic gredient descent(batch-size=1)

* Lose speedup from vectorization

##### In between(batch-size not too big(small))

* Faster learning
* vectorization(~1000)
* make progress without needing to wait processing entire dataset

##### Batch gredient descent(batch-size=m)

* Too long per iteration



#### Choosing your mini-batch size

* If small train set(m<=2000) : use batch gradient descent
* typical minibatch-size :  64, 128, 256, 512 (power of 2)
* Make sure minibatch fit in CPU/GPU memory 







## Exponentially weighted averages



## Understanding exponentially weighted averages

 v_100 = 0.1 theta_100 + 0.1 theta+_99  + ....                                                                



##### Implmenting exponentially weighted averages

V := 0

V := Beta * v + (1 - Beta) * theta_1

V := Beta * v + (1 - Beta) * theta_2

