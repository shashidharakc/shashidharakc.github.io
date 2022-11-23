## **How Weights are updated during BackProp**


In this Blog/ notebook we will discuss how weights of the network are updated/ adjusted in a simple classification model. For this we are considering Iris dataset which has 3 class, 4 features for each data point. We will build our netowrk using JAX both Stax api and custom model, we will use Categorical Cross entropy loss with softmax as final layer, with no hidden layer. No hidden layer cos of simplicity of network, we are not building a model that solves the problem with great accuracy, but we just want to understand how weights are adjusted based on the error. By the end of this blog we should have a better understanding of how the weights are updated in classification model.

Architicture of the model is as shown below diagram.

Below is the same diagram updated with some variables/constants updates with random values.

Apart from the inputs and weights we have shown in the figure above, lets say we have True Value = [1.0, 0.0, 0.0] for the give data points. Keeping these values lets go through step by step on how we update the weights.

## Forward Prop

Now Lets go through a step of forward pass. What we have so far is input data, ***X*** nd randomly initiallized weights ***W***. Again for making the calcuation easy and of easy of understanding we will ommit the bias and procced. 

Lets Calculate, this is summation of product of input and weights, generally.

<img src="https://render.githubusercontent.com/render/math?math=\sum_{n=0}^\infty\frac{1}{2^n}">

<img src="https://render.githubusercontent.com/render/math?math=a^{2} %2B b^{2} = c^{2}">

```math
e^{i\tau} - 1 = 0
```

'$Z_{1}^{1}$'
