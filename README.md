## **How Weights are updated during BackProp**


In this Blog/ notebook we will discuss how weights of the network are updated/ adjusted in a simple classification model. For this we are considering Iris dataset which has 3 class, 4 features for each data point. We will build our netowrk using JAX both Stax api and custom model, we will use Categorical Cross entropy loss with softmax as final layer, with no hidden layer. No hidden layer cos of simplicity of network, we are not building a model that solves the problem with great accuracy, but we just want to understand how weights are adjusted based on the error. By the end of this blog we should have a better understanding of how the weights are updated in classification model.

Architicture of the model is as shown below diagram.

Below is the same diagram updated with some variables/constants updates with random values.

Apart from the inputs and weights we have shown in the figure above, lets say we have True Value = [1.0, 0.0, 0.0] for the give data points. Keeping these values lets go through step by step on how we update the weights.

## Forward Prop

Now Lets go through a step of forward pass. What we have so far is input data, ***X*** nd randomly initiallized weights ***W***. Again for making the calcuation easy and of easy of understanding we will ommit the bias and procced. 

Lets Calculate, this is summation of product of input and weights, generally.

<img src="https://render.githubusercontent.com/render/math?math={\color{black} \displaystyle\f(x)=sin(x)}">

![formula](https://render.githubusercontent.com/render/math?math=\color{red}\sum_{d=0}^{d_{max}})

so, 

<img src="https://render.githubusercontent.com/render/math?math={\color{black} \displaystyle\Z_{1}^{1} = x^{1} * w_{11}^{1}}">

![formula](https://render.githubusercontent.com/render/math?math=\color{red}\large\Z_{1}^{1} = x^{1} * w_{11}^{1})


When we plug in the number which yields 

![formula](https://render.githubusercontent.com/render/math?math=\color{red}\large\Z = \sum_{i} = x_{i}*w{i})
