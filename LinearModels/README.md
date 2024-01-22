# Classification 
## Gradient Descent 

## Mirror Descent 
Mirror Descent for Classification with exponential or logistic loss was investigated in ["A Unified Approach to Controlling Implicit Regularization via Mirror Descent"](https://arxiv.org/pdf/2306.13853v1.pdf). 
### Growth Rate of Parameters  
Due to the exponential loss not having a global minimum for separable datasets, the parameters drift off to infinity. 
Here we visualize the growth rate of the parameter $p$-norms during training under $p$-Gradient Descent for various values of $p$. The implementation of $p$-Gradient Descent can be found in [Optimizers/MirrorDescent](https://github.com/fekonrad/Implicit-Bias/blob/main/Optimizers/MirrorDescent.py).
Note that it was shown in ["The Implicit Bias of Gradient Descent on Separable Data"](https://arxiv.org/pdf/1710.10345.pdf) Theorem 3 and in ["A Unified Approach to Controlling Implicit Regularization via Mirror Descent"](https://arxiv.org/pdf/2306.13853v1.pdf)
Lemma 18 that the growth rate should be roughly logarithmic in the number of epochs. 

![](ParameterGrowthLinearModel.png)

**Note: There seem to be some numerical issues when training with $p$-Gradient Descent for large values of $p$, as the parameters seem to shoot to infinity at the start of training.** 
