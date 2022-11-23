## **Back Propagation In Cross Entropy Loss with Softmax**


In this Blog/ notebook we will discuss how weights of the network are updated/ adjusted in a simple classification model. For this we are considering Iris dataset which has 3 class, 4 features for each data point. We will build our netowrk using JAX both Stax api and custom model, we will use Categorical Cross entropy loss with softmax as final layer, with no hidden layer. No hidden layer cos of simplicity of network, we are not building a model that solves the problem with great accuracy, but we just want to understand how weights are adjusted based on the error. By the end of this blog we should have a better understanding of how the weights are updated in classification model.

Architicture of the model is as shown below diagram.

Below is the same diagram updated with some variables/constants updates with random values.

Apart from the inputs and weights we have shown in the figure above, lets say we have True Value = [1.0, 0.0, 0.0] for the give data points. Keeping these values lets go through step by step on how we update the weights.

## Forward Prop

Now Lets go through forward pass. What we have so far is input data, ***X*** nd randomly initiallized weights ***W***, easy of understanding we will ommit the bias and procced. 

Lets Calculate $Z_{1}^{1}$, this is summation of product of input and weights, generally  $Z = \sum_{i} = x_{i} * w_{i} $. 

so, $Z_{1}^{1} = x^{1} * w_{11}^{1} + x^{2} * w_{12}^{1} + x^{3} * w_{13}^{1} + x_{4} * w_{14}^{1}$

When we plug in the number which yields 

$Z_{1}^{1} = (0.81 * 0.15) + (0.57 * 0.29) + (0.28 * 0.10) + (0.84 * 0.71) = 0.91119$

similarly $Z_{2}^{1} = 0.6105, Z_{3}^{1} = -0.1517$

Now we calculate sofmax of the $Z$ ouput using below equation 

$softmax(Y_{i}) = \frac {e^{y_{i}}}{\sum_{j=1}^{J} e^{y_{j}} }$

Plugging the logits i.e $Z$ to the above softamx equation we get 

$0.4794433, 0.35493179, 0.16562491$

Categorical cross-entropy loss is give by below equation, lets calculate the loss using all the data we have so far.

$ CrosEntropyLoss = \sum_{h} y_h log(\hat y_h)$

$ E(\theta) = \frac{1}{m} \sum_{i=1}^{m} L(y, \hat y)$

$ L_{CE} = - \sum_{i=1} T_i log(S_i)$

Lets plugin the numbers we have and calculate the loss.

$        = -[1.0 * log_2(0.4794433) + 0.0 * log_2( 0.35493179) + 0.0 * log_2(0.16562491)] $

$        = - log_2(0.4794433) $

$        = 1.0605678 $


<img src="https://render.githubusercontent.com/render/math?math=\sum_{n=0}^\infty\frac{1}{2^n}">

<img src="https://render.githubusercontent.com/render/math?math=a^{2} %2B b^{2} = c^{2}">

```math
`$e^{i\tau} - 1 = 0$`
```

`$Z_{1}^{1}$`


$`Z_{1}^{1}`$


$$
f(a) = \frac{1}{2\pi i} \oint_{\gamma}\frac{f(z)}{z-a} dz.
$$
